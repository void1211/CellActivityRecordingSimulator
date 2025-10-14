"""
GPU対応ユーティリティモジュール
CuPyとNumbaを使用してGPU演算を提供する
"""

import numpy as np
from typing import Optional, Union
import logging

# GPU対応のためのインポート（オプショナル）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPyが利用できません。CPUモードで実行します。")

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numbaが利用できません。CPUモードで実行します。")

class GPUManager:
    """GPU管理クラス"""
    
    def __init__(self):
        self.device = None
        self.gpu_available = False
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """GPUの初期化"""
        if CUPY_AVAILABLE:
            try:
                # GPUデバイスを取得
                self.device = cp.cuda.Device(0)
                self.device.use()
                self.gpu_available = True
                logging.info(f"GPU使用可能: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
            except Exception as e:
                logging.warning(f"GPU初期化に失敗: {e}")
                self.gpu_available = False
        else:
            logging.info("CuPyが利用できないため、CPUモードで実行します")
    
    def is_gpu_available(self) -> bool:
        """GPUが利用可能かどうかを返す"""
        return self.gpu_available
    
    def get_device_info(self) -> dict:
        """デバイス情報を取得"""
        if not self.gpu_available:
            return {"device": "CPU", "available": False}
        
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            return {
                "device": "GPU",
                "name": props['name'].decode(),
                "memory": props['totalGlobalMem'],
                "available": True
            }
        except Exception as e:
            logging.error(f"デバイス情報の取得に失敗: {e}")
            return {"device": "CPU", "available": False}

# グローバルGPUマネージャー
gpu_manager = GPUManager()

def get_array_module(xp=None):
    """適切な配列モジュールを返す（NumPyまたはCuPy）"""
    if xp is not None:
        return xp
    
    if gpu_manager.is_gpu_available():
        return cp
    else:
        return np

def to_gpu(array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray']:
    """配列をGPUに転送"""
    if gpu_manager.is_gpu_available() and isinstance(array, np.ndarray):
        return cp.asarray(array)
    return array

def to_cpu(array) -> np.ndarray:
    """配列をCPUに転送"""
    if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

def add_spikes_gpu(signal: np.ndarray, spike_times: list, spike_template: np.ndarray, spike_amplitudes: list) -> np.ndarray:
    """GPU対応のスパイク追加関数"""
    xp = get_array_module()
    
    # 配列を適切なデバイスに転送
    signal_gpu = xp.asarray(signal)
    template_gpu = xp.asarray(spike_template)
    
    # ピーク位置を計算
    if xp.min(template_gpu) < 0 and abs(xp.min(template_gpu)) > abs(xp.max(template_gpu)):
        peak = int(xp.argmin(template_gpu))
    else:
        peak = int(xp.argmax(template_gpu))
    
    # スパイクを追加
    for spike_time, spike_amp in zip(spike_times, spike_amplitudes):
        start = int(spike_time - peak)
        end = int(start + len(template_gpu))
        
        if 0 <= start and end <= len(signal_gpu):
            signal_gpu[start:end] += spike_amp * template_gpu
    
    return to_cpu(signal_gpu)

def _add_spikes_cpu(signal: np.ndarray, spike_times: np.ndarray, spike_template: np.ndarray, spike_amplitudes: np.ndarray) -> np.ndarray:
    """CPU版のスパイク追加関数"""
    # ピーク位置を計算
    if np.min(spike_template) < 0 and abs(np.min(spike_template)) > abs(np.max(spike_template)):
        peak = np.argmin(spike_template)
    else:
        peak = np.argmax(spike_template)
    
    # スパイクを追加
    for i in range(len(spike_times)):
        spike_time = int(spike_times[i])
        spike_amp = spike_amplitudes[i]
        
        start = spike_time - peak
        end = start + len(spike_template)
        
        if 0 <= start and end <= len(signal):
            for j in range(len(spike_template)):
                signal[start + j] += spike_amp * spike_template[j]
    
    return signal

# Numbaが利用可能な場合はJITコンパイルを適用
if NUMBA_AVAILABLE:
    from numba import jit
    add_spikes_numba = jit(nopython=True, cache=True)(_add_spikes_cpu)
else:
    add_spikes_numba = _add_spikes_cpu

def generate_unit_spike_signal_gpu(settings: dict, spike_times: list, spike_template: np.ndarray, spike_amplitudes: list) -> np.ndarray:
    """GPU対応のUnitスパイク信号生成"""
    duration_samples = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
    
    # GPUが利用可能な場合はGPU版を使用
    if gpu_manager.is_gpu_available():
        signal = add_spikes_gpu(np.zeros(duration_samples, dtype=np.float64), spike_times, spike_template, spike_amplitudes)
    else:
        # CPU版（Numba JITを使用）
        signal = add_spikes_numba(
            np.zeros(duration_samples, dtype=np.float64),
            np.array(spike_times, dtype=np.int32),
            np.array(spike_template, dtype=np.float64),
            np.array(spike_amplitudes, dtype=np.float64)
        )
    
    return signal

def calculate_distance_gpu(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """GPU対応の距離計算"""
    xp = get_array_module()
    
    pos1_gpu = xp.asarray(pos1)
    pos2_gpu = xp.asarray(pos2)
    
    distance_squared = xp.sum((pos1_gpu - pos2_gpu) ** 2)
    distance = float(xp.sqrt(distance_squared))
    
    return distance

def apply_attenuation_gpu(signal: np.ndarray, distance: float, attenuation_constant: float) -> np.ndarray:
    """GPU対応の減衰適用"""
    xp = get_array_module()
    
    signal_gpu = xp.asarray(signal)
    attenuation_factor = xp.exp(-distance / attenuation_constant)
    
    return to_cpu(signal_gpu * attenuation_factor)
