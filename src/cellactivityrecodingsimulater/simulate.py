import numpy as np
import logging

from Cell import Cell
from Site import Site
from calculate import calcScaledSpikeAmp, calcDistance, gabor
from spikeinterface.core.generate import generate_single_fake_waveform


def simulateBackgroundActivity(duration: float, fs: float, noise_cells: list[Cell], site: Site, attenTime: float) -> list[float]:
    """
    ノイズ細胞の活動を記録サイトの信号に追加する
    
    Args:
        noise_cells: ノイズ細胞のリスト
        sites: 記録サイトのリスト
        settings: シミュレーション設定
    """
    if not noise_cells:
        logging.warning("ノイズ細胞が存在しません")
        return
    
    # サイトの信号をnumpy配列に変換
    site_signal = np.zeros(int(duration * fs))
    
    added_spikes = 0
    for cell_idx, cell in enumerate(noise_cells):
        # 各細胞からの信号を計算
        scaled_amps = calcScaledSpikeAmp(cell.spikeAmpList, calcDistance(cell, site), attenTime)
        
        # スパイクを信号に追加
        spikeTimes = cell.spikeTimeList
        spikeTemp = cell.spikeTemp
        peak = np.argmax(np.abs(spikeTemp))
        
        cell_added_spikes = 0
        for spikeTime, spikeAmp in zip(spikeTimes, scaled_amps):
            start = int(spikeTime - peak)
            end = int(start + len(spikeTemp))
            if not (0 <= start and end <= len(site_signal)):
                continue
            site_signal[start:end] += spikeAmp * spikeTemp
            cell_added_spikes += 1
        
        if cell_added_spikes > 0:
            added_spikes += cell_added_spikes
            if cell_idx < 5:  # 最初の5個の細胞のみログ出力
                distance = calcDistance(cell, site)
                logging.debug(f"  細胞{cell_idx}: 距離={distance:.2f}μm, 追加スパイク数={cell_added_spikes}")
    
    return site_signal.tolist()

def simulateSpikeTimes(duration:float, fs:float, rate:float, isRefractory:bool=False, refractoryPeriod:float=0) -> list[int]:
    """セルのスパイク時間をシミュレートする"""

    # より現実的な不応期モデル
    if isRefractory:
        # 絶対不応期のみを考慮
        absolute_refractory = refractoryPeriod * 1.0  # 絶対不応期（100%）
        
        # スパイク時間を生成
        spike_times = []
        current_time = 0
        
        while current_time < duration * fs:
            # 絶対不応期中はスパイクを発火できない
            current_time += absolute_refractory * fs / 1000
            
            # 通常のスパイク間隔を生成
            if current_time < duration * fs:
                # 指数分布でスパイク間隔を生成
                isi = np.random.exponential(1 / rate) 
                current_time += isi * fs
                
                if current_time < duration * fs:
                    spike_times.append(int(current_time))
        
        return spike_times
    else:
        # 不応期なしの場合（従来の実装）
        isi = np.random.exponential(1 / rate, size=1000)
        isi = np.ceil(isi * fs)
        spikeTimes = np.cumsum(isi)
        spikeTimes = spikeTimes[spikeTimes < int(duration * fs)]
        return spikeTimes

def simulateRandomNoise(duration: float, fs: float, noiseType: str="normal", noiseAmp: float=1.0) -> list[float]:
    """ランダムノイズをシミュレートする"""

    if noiseType == "normal":
        noise = np.random.default_rng().integers(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
    elif noiseType == "gaussian":
        noise = np.random.normal(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
    else:
        raise ValueError(f"Invalid noise type: {noiseType}")

    return noise

def simulateSpikeTemplate(fs: float, spikeType: str, randType: str, spikeWidth: float,
    gaborSigma: list[float]=None, gaborf0: list[float]=None, gabortheta: list[float]=None, 
    ms_before: list[float]=None, ms_after: list[float]=None, negative_amplitude: list[float]=None,
    positive_amplitude: list[float]=None, depolarization_ms: list[float]=None,
    repolarization_ms: list[float]=None, recovery_ms: list[float]=None, smooth_ms: list[float]=None,
    ) -> list[float]:
    """スパイクテンプレートをシミュレートする"""
    if spikeType == "gabor":
        if randType == "list":
            gaborSigma = np.random.choice(gaborSigma)
            gaborf0 = np.random.choice(gaborf0)
            gabortheta = np.random.choice(gabortheta)
        elif randType == "range":
            gaborSigma = np.random.uniform(gaborSigma[0], gaborSigma[1])
            gaborf0 = np.random.uniform(gaborf0[0], gaborf0[1])
            gabortheta = np.random.uniform(gabortheta[0], gabortheta[1])
        else:
            raise ValueError(f"randType '{randType}' is not supported for template simulation")
        
        gabortheta_rad = gabortheta * np.pi / 180
        spikeTemplate = gabor(gaborSigma, gaborf0, gabortheta_rad, fs, spikeWidth)
        return spikeTemplate

    elif spikeType == "exponential":
        if randType == "list":
            ms_before = np.random.choice(ms_before)
            ms_after = np.random.choice(ms_after)
            negative_amplitude = np.random.choice(negative_amplitude)
            positive_amplitude = np.random.choice(positive_amplitude)
            depolarization_ms = np.random.choice(depolarization_ms)
            repolarization_ms = np.random.choice(repolarization_ms)
            recovery_ms = np.random.choice(recovery_ms)
            smooth_ms = np.random.choice(smooth_ms)
        elif randType == "range":
            ms_before = np.random.uniform(ms_before[0], ms_before[1]) if len(ms_before) > 1 else ms_before[0]
            ms_after = np.random.uniform(ms_after[0], ms_after[1]) if len(ms_after) > 1 else ms_after[0]
            negative_amplitude = np.random.uniform(negative_amplitude[0], negative_amplitude[1]) if len(negative_amplitude) > 1 else negative_amplitude[0]
            positive_amplitude = np.random.uniform(positive_amplitude[0], positive_amplitude[1]) if len(positive_amplitude) > 1 else positive_amplitude[0]
            depolarization_ms = np.random.uniform(depolarization_ms[0], depolarization_ms[1]) if len(depolarization_ms) > 1 else depolarization_ms[0]
            repolarization_ms = np.random.uniform(repolarization_ms[0], repolarization_ms[1]) if len(repolarization_ms) > 1 else repolarization_ms[0]
            recovery_ms = np.random.uniform(recovery_ms[0], recovery_ms[1]) if len(recovery_ms) > 1 else recovery_ms[0]
            smooth_ms = np.random.uniform(smooth_ms[0], smooth_ms[1]) if len(smooth_ms) > 1 else smooth_ms[0]
        else:
            raise ValueError(f"randType '{randType}' is not supported for template simulation")

        spikeTemplate = simulateExponentialTemplate(
            fs,
            ms_before,
            ms_after,
            negative_amplitude,
            positive_amplitude,
            depolarization_ms,
            repolarization_ms,
            recovery_ms,
            smooth_ms)
        
        return spikeTemplate
    else:
        raise ValueError(f"spikeType '{spikeType}' is not supported for template simulation")

def simulateExponentialTemplate(
        fs: float, 
        ms_before: float, 
        ms_after: float, 
        negative_amplitude: float, 
        positive_amplitude: float, 
        depolarization_ms: float, 
        repolarization_ms: float, 
        recovery_ms: float, 
        smooth_ms: float) -> list[float]:
    
    spikeTemplate = generate_single_fake_waveform(
        sampling_frequency=fs,
        ms_before=ms_before,
        ms_after=ms_after,
        negative_amplitude=negative_amplitude,
        positive_amplitude=positive_amplitude,
        depolarization_ms=depolarization_ms,
        repolarization_ms=repolarization_ms,
        recovery_ms=recovery_ms,
        smooth_ms=smooth_ms)    
    # ピークを絶対値１に調整
    spikeTemplate = spikeTemplate / np.max(np.abs(spikeTemplate))
    return spikeTemplate

def simulateDrift(duration: float, fs: float, driftType: str = "linear", drift_amplitude: float = 50.0, drift_frequency: float = 0.1) -> np.ndarray:
    """ドリフト信号をシミュレートする"""
    
    signal_length = int(duration * fs)
    
    if driftType == "linear":
        # 線形ドリフト（時間とともに直線的に変化）
        t = np.linspace(0, duration, signal_length)
        drift = np.linspace(-drift_amplitude/2, drift_amplitude/2, signal_length)
        
    elif driftType == "exponential":
        # 指数関数的ドリフト（時間とともに指数関数的に変化）
        t = np.linspace(0, duration, signal_length)
        drift = drift_amplitude * (np.exp(-t / (duration / 3)) - 0.5)
        
    elif driftType == "oscillatory":
        # 振動的ドリフト（正弦波的な変化）
        t = np.linspace(0, duration, signal_length)
        drift = drift_amplitude * np.sin(2 * np.pi * drift_frequency * t)
        
    elif driftType == "random_walk":
        # ランダムウォークドリフト（ランダムな歩行）
        steps = np.random.normal(0, drift_amplitude / 100, signal_length)
        drift = np.cumsum(steps)
        # 振幅を制限
        drift = drift - np.mean(drift)
        drift = drift * (drift_amplitude / np.max(np.abs(drift)))
        
    elif driftType == "step":
        # ステップ状ドリフト（段階的な変化）
        drift = np.zeros(signal_length)
        step_times = np.random.choice(signal_length, size=3, replace=False)
        step_times.sort()
        
        current_level = 0
        for i, step_time in enumerate(step_times):
            if i < len(step_times) - 1:
                next_step = step_times[i + 1]
                drift[step_time:next_step] = current_level
                current_level += np.random.uniform(-drift_amplitude/2, drift_amplitude/2)
            else:
                drift[step_time:] = current_level
                
    else:
        raise ValueError(f"Unknown drift type: {driftType}")
    
    return drift.astype(np.float64)

def simulatePowerLineNoise(duration: float, fs: float, powerLineFreq: float = 50.0, power_noise_amplitude: float = 20.0) -> np.ndarray:
    """
    電源ノイズ（50Hz/60Hz）をシミュレートする
    
    Args:
        settings: シミュレーション設定
        powerLineFreq: 電源周波数（Hz、デフォルト50Hz）
    
    Returns:
        np.ndarray: 電源ノイズ信号
    """
    signal_length = int(duration * fs)
    
    # 時間軸
    t = np.linspace(0, duration, signal_length)
    
    # 基本周波数の電源ノイズ
    power_noise = power_noise_amplitude * np.sin(2 * np.pi * powerLineFreq * t)
    
    # 高調波を追加（3次、5次、7次）
    harmonics = [3, 5, 7]
    harmonic_amplitudes = [0.3, 0.2, 0.1]  # 基本波に対する比率
    
    for harmonic, amplitude_ratio in zip(harmonics, harmonic_amplitudes):
        harmonic_noise = power_noise_amplitude * amplitude_ratio * np.sin(2 * np.pi * powerLineFreq * harmonic * t)
        power_noise += harmonic_noise
    
    # 位相のゆらぎを追加（より現実的な電源ノイズ）
    phase_noise = np.random.normal(0, 0.1, signal_length)  # 小さな位相ノイズ
    power_noise_with_phase = power_noise_amplitude * np.sin(2 * np.pi * powerLineFreq * t + phase_noise)
    
    # 基本ノイズと位相ノイズを組み合わせ
    final_power_noise = 0.7 * power_noise + 0.3 * power_noise_with_phase
    
    return final_power_noise.astype(np.float64)