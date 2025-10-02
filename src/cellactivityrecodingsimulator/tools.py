from pathlib import Path
import logging
import numpy as np
from scipy.signal import butter, filtfilt

from .Settings import Settings

def make_save_dir(pathSaveDir: Path) -> Path:
    """保存ディレクトリを作成する"""
    Path(pathSaveDir).mkdir(parents=True, exist_ok=True)
    # if pathSaveDir.exists() and any(pathSaveDir.iterdir()):
        # user_input = input(f"保存先ディレクトリ {pathSaveDir} はすでに存在し、ファイルが含まれています。上書きしますか? (y/n): ")
        # if user_input.lower() != 'y':
        #     # 別名で保存ディレクトリを作成
        #     counter = 1
        #     while True:
        #         new_path = pathSaveDir.parent / f"{pathSaveDir.name}_{counter}"
        #         if not new_path.exists():
        #             pathSaveDir = new_path
        #             pathSaveDir.mkdir(parents=True)
        #             logging.info(f"別名で保存ディレクトリを作成: {pathSaveDir}")
        #             break
        #         counter += 1
    return Path(pathSaveDir)    

def addSpikeToSignal(signal: np.ndarray, spikeTimes: list[int], spikeTemp: list[float], SpikeAmpList: list[float]) -> np.ndarray:
    """スパイクを信号に追加する"""
    
    # 信号をnumpy配列に変換
    if isinstance(signal, list):
        signal = np.array(signal, dtype=np.float64)
    else:
        # 信号を浮動小数点型に変換 
        if signal.dtype != np.float64:
            signal = signal.astype(np.float64)
    
    # ピーク位置を正しく計算（負のピークも考慮）
    if np.min(spikeTemp) < 0 and abs(np.min(spikeTemp)) > abs(np.max(spikeTemp)):
        # 負のピークが主成分の場合
        peak = np.argmin(spikeTemp)
    else:
        # 正のピークが主成分の場合
        peak = np.argmax(spikeTemp)
    
    for spikeTime, spikeAmp in zip(spikeTimes, SpikeAmpList):
        start = int(spikeTime - peak)
        end = int(start + len(spikeTemp))
        if not (0 <= start and end <= len(signal)):
            continue
        signal[start:end] += spikeAmp * spikeTemp
    return signal

def filterSignal(signal: np.ndarray, fs: float, lowCutoffFreq: float, highCutoffFreq: float) -> np.ndarray:
    """信号をバンドパスフィルタリングする"""
    # 信号の長さをチェック
    min_length = 50  # 最小信号長（フィルタリングに必要な最小サンプル数）
    if len(signal) < min_length:
        logging.warning(f"信号が短すぎます（{len(signal)}サンプル）。フィルタリングをスキップします。")
        return signal
    
    try:
        b, a = butter(2, [lowCutoffFreq / (fs / 2), highCutoffFreq / (fs / 2)], btype='bandpass')
        filteredSignal = filtfilt(b, a, signal)
        return filteredSignal
    except ValueError as e:
        logging.warning(f"フィルタリングでエラーが発生しました: {e}。元の信号を返します。")
        return signal

def convert_legacySettings(legacySettings: dict) -> dict:
    """レガシーな設定を新しい設定に変換する"""
    def safe_get(key: str, default=None):
        """安全にキーを取得する"""
        return legacySettings.get(key, default)
    
    newSettings = {
        "baseSettings": {
            "name": safe_get("name"),
            "pathSaveDir": safe_get("pathSaveDir"),
            "fs": safe_get("fs"),
            "duration": safe_get("duration"),
            "random_seed": safe_get("random_seed"),
        },
        "spikeSettings": {
            "rate": safe_get("avgSpikeRate"),
            "isRefractory": safe_get("isRefractory"),
            "refractoryPeriod": safe_get("refractoryPeriod"),
            "absolute_refractory_ratio": safe_get("absolute_refractory_ratio", 1.0),
            "amplitudeMax": safe_get("spikeAmpMax"),
            "amplitudeMin": safe_get("spikeAmpMin"),
            "attenTime": safe_get("attenTime"),
            "spikeType": safe_get("spikeType"),
            "gabor": {
                "randType": safe_get("randType"),
                "sigma": safe_get("sigma"),
                "f0": safe_get("f0"),
                "theta": safe_get("theta"),
                "width": safe_get("spikeWidth"),
            },
            "exponential": {
                "randType": safe_get("randType"),
                "ms_before": safe_get("ms_before"),
                "ms_after": safe_get("ms_after"),
                "negative_amplitude": safe_get("negative_amplitude"),
                "positive_amplitude": safe_get("positive_amplitude"),
                "depolarization_ms": safe_get("depolarization_ms"),
                "repolarization_ms": safe_get("repolarization_ms"),
                "recovery_ms": safe_get("recovery_ms"),
                "smooth_ms": safe_get("smooth_ms"),
            },
            "template": {
                "pathSpikeList": safe_get("pathSpikeList"),
            },
            "truth": {
                "pathSpikeList": safe_get("pathSpikeList"),
            },
        },
        "noiseSettings": {
            "noiseType": safe_get("noiseType"),
            "model": {
                "density": safe_get("density"),
                "margin": safe_get("margin"),
                "inviolableArea": safe_get("inviolableArea"),
            },
            "normal": {
                "amplitude": safe_get("amplitude"),
            },
            "gaussian": {
                "amplitude": safe_get("amplitude"),
                "location": safe_get("loc"),
                "scale": safe_get("scale"),
            },
            "truth": {
                "pathNoise": safe_get("pathNoise"),
                "pathSites": safe_get("pathSites"),
            },
        },
        "driftSettings": {
            "enable": safe_get("enable_drift"),
            "driftType": safe_get("drift_type"),
            "random_walk": {
                "amplitude": safe_get("drift_amplitude"),
                "frequency": safe_get("drift_frequency"),
            },
            "step": {
                "amplitude": safe_get("drift_amplitude"),
                "frequency": safe_get("drift_frequency"),
            },
            "oscillatory": {
                "amplitude": safe_get("drift_amplitude"),
                "frequency": safe_get("drift_frequency"),
            },
            "exponential": {
                "amplitude": safe_get("drift_amplitude"),
                "frequency": safe_get("drift_frequency"),
            },
        },
        "powerNoiseSettings": {
            "enable": safe_get("enable_power_noise"),
            "frequency": safe_get("power_line_frequency"),
            "amplitude": safe_get("power_noise_amplitude"),
        },
        "templateSimilarityControlSettings": {
            "enable": safe_get("enable_template_similarity_control"),
            "min_cosine_similarity": safe_get("min_cosine_similarity"),
            "max_cosine_similarity": safe_get("max_cosine_similarity"),
            "similarity_control_attempts": safe_get("similarity_control_attempts"),
        },
    }

    return newSettings