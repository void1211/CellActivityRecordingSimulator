from pathlib import Path
import logging
import numpy as np
from scipy.signal import butter, filtfilt
import random

from Cell import Cell
from Site import Site
from Settings import Settings

def getProjectRoot() -> Path:
    """プロジェクトのルートディレクトリを取得する"""
    return Path(__file__).resolve().parents[2]

def makeSaveDir(pathSaveDir: Path):
    """保存ディレクトリを作成する"""
    pathSaveDir.mkdir(parents=True, exist_ok=True)
    if pathSaveDir.exists() and any(pathSaveDir.iterdir()):
        user_input = input(f"保存先ディレクトリ {pathSaveDir} はすでに存在し、ファイルが含まれています。上書きしますか? (y/n): ")
        if user_input.lower() != 'y':
            # 別名で保存ディレクトリを作成
            counter = 1
            while True:
                new_path = pathSaveDir.parent / f"{pathSaveDir.name}_{counter}"
                if not new_path.exists():
                    pathSaveDir = new_path
                    pathSaveDir.mkdir(parents=True)
                    logging.info(f"別名で保存ディレクトリを作成: {pathSaveDir}")
                    break
                counter += 1
            return

def generateNoiseCells(settings: Settings, sites: list[Site]) -> list[Cell]:
    """
    3次元空間上に背景ノイズを生成する細胞を配置する
    
    Args:
        settings: シミュレーション設定
        sites: 記録サイトのリスト
    
    Returns:
        list[Cell]: 生成されたノイズ細胞のリスト
    """
    if not settings.noiseType == "model":
        return []
    
    # 乱数シードを設定
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)
        random.seed(settings.random_seed)
    
    # 記録サイトの範囲を計算
    site_x_coords = [site.x for site in sites]
    site_y_coords = [site.y for site in sites]
    site_z_coords = [site.z for site in sites]
    
    min_x, max_x = min(site_x_coords), max(site_x_coords)
    min_y, max_y = min(site_y_coords), max(site_y_coords)
    min_z, max_z = min(site_z_coords), max(site_z_coords)
    
    # マージンを追加して配置範囲を拡張
    margin = settings.margin
    volume_x = (max_x - min_x) + 2 * margin
    volume_y = (max_y - min_y) + 2 * margin
    volume_z = (max_z - min_z) + 2 * margin
    
    # 体積をmm³に変換（μm³ → mm³）
    volume_mm3 = (volume_x * volume_y * volume_z) / (1000**3)
    
    # 細胞数を計算（密度 × 体積）
    cell_count = int(settings.cell_density * volume_mm3)
    
    # 細胞をランダムに配置
    noise_cells = []
    for i in range(cell_count):
        # ランダムな位置を生成
        x = np.random.uniform(min_x - margin, max_x + margin)
        y = np.random.uniform(min_y - margin, max_y + margin)
        z = np.random.uniform(min_z - margin, max_z + margin)
        
        # 細胞を生成
        cell = Cell(
            id=i,
            x=int(x),
            y=int(y),
            z=int(z)
        )
        
        # スパイク活動をシミュレート
        cell.spikeTimeList = simulateSpikeTimes(settings)
        cell.spikeAmpList = [calcSpikeAmp(settings) for _ in cell.spikeTimeList]
        cell.spikeTemp = simulateSpikeTemplate(settings)
        
        noise_cells.append(cell)
    
    return noise_cells

def addNoiseCellsToSite(noise_cells: list[Cell], site: Site, settings: Settings) -> list[float]:
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
    site_signal = np.zeros(int(settings.duration * settings.fs))
    
    added_spikes = 0
    for cell_idx, cell in enumerate(noise_cells):
        # 各細胞からの信号を計算
        scaled_amps = calcScaledSpikeAmp(cell, site, settings)
        
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

def simulateSpikeTimes(settings: Settings) -> list[int]:
    """セルのスパイク時間をシミュレートする"""
    duration = settings.duration
    fs = settings.fs
    avgSpikeRate = settings.avgSpikeRate

    if settings.isRefractory:
        refractoryPeriod = settings.refractoryPeriod
    else:
        refractoryPeriod = 0

    # より現実的な不応期モデル
    if refractoryPeriod > 0:
        # 絶対不応期 + 相対不応期を考慮
        absolute_refractory = refractoryPeriod * 0.4  # 絶対不応期（40%）
        relative_refractory = refractoryPeriod * 0.6   # 相対不応期（60%）
        
        # スパイク時間を生成
        spike_times = []
        current_time = 0
        
        while current_time < duration * fs:
            # 絶対不応期中はスパイクを発火できない
            current_time += absolute_refractory * fs / 1000
            
            # 相対不応期中は確率的にスパイクを発火
            if current_time < duration * fs:
                # 相対不応期中の発火確率を計算
                relative_prob = 0.3  # 相対不応期中の発火確率（30%）
                
                if np.random.random() < relative_prob:
                    # 相対不応期中にスパイクを発火
                    current_time += relative_refractory * fs / 1000
                else:
                    # 相対不応期をスキップ
                    current_time += relative_refractory * fs / 1000
            
            # 通常のスパイク間隔を生成
            if current_time < duration * fs:
                # 指数分布でスパイク間隔を生成
                isi = np.random.exponential(1 / avgSpikeRate)
                current_time += isi * fs
                
                if current_time < duration * fs:
                    spike_times.append(int(current_time))
        
        return spike_times
    else:
        # 不応期なしの場合（従来の実装）
        isi = np.random.exponential(1 / avgSpikeRate, size=1000) + refractoryPeriod / 1000
        isi = np.ceil(isi * fs)
        spikeTimes = np.cumsum(isi)
        spikeTimes = spikeTimes[spikeTimes < int(duration * fs)]
        return spikeTimes

def simulateRecordingNoise(settings: Settings, noiseType: str) -> list[float]:
    """録音ノイズをシミュレートする"""
    duration = settings.duration
    fs = settings.fs
    
    if noiseType in ["normal", "gaussian"]:
        if settings.noiseAmp is None:
            raise ValueError(f"noiseAmp is required when noiseType is {noiseType}")
        noiseAmp = settings.noiseAmp
    else:
        raise ValueError(f"Invalid noise type: {noiseType}")

    if noiseType == "normal":
        noise = np.random.default_rng().integers(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
    elif noiseType == "gaussian":
        noise = np.random.normal(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
    else:
        raise ValueError(f"Invalid noise type: {noiseType}")

    return noise

def getRecordingNoiseFromTruth(settings: Settings) -> list[float]:
    """真の録音ノイズを取得する"""
    if settings.pathTruthNoise is None:
        raise ValueError("pathTruthNoise is required when noiseType is 'truth'")
    
    try:
        noise = np.load(settings.pathTruthNoise).astype(np.float64)
        return noise
    except FileNotFoundError:
        noise = np.zeros(int(settings.duration * settings.fs)).astype(np.float64)
        logging.warning(f"File not found: {settings.pathTruthNoise}")
        return noise
        

def addSpikeToSignal(cell: Cell, site: Site, scaledSpikeAmpList: list[float]) -> list[float]:
    """スパイクを信号に追加する"""
    signal = site.signalRaw
    
    # 信号をnumpy配列に変換
    if isinstance(signal, list):
        signal = np.array(signal, dtype=np.float64)
    else:
        # 信号を浮動小数点型に変換
        if signal.dtype != np.float64:
            signal = signal.astype(np.float64)
    
    spikeTimes = cell.spikeTimeList
    spikeTemp = cell.spikeTemp
    peak = np.argmax(np.abs(spikeTemp))
    for spikeTime, spikeAmp in zip(spikeTimes, scaledSpikeAmpList):
        start = int(spikeTime - peak)
        end = int(start + len(spikeTemp))
        if not (0 <= start and end <= len(signal)):
            continue
        signal[start:end] += spikeAmp * spikeTemp
    site.signalRaw = signal.tolist()
    return signal.tolist()
    
def calcSpikeAmp(settings: Settings) -> float:
    """スパイク振幅を計算する"""
    if settings.spikeAmpMax is None or settings.spikeAmpMin is None:
        raise ValueError("spikeAmpMax and spikeAmpMin are required for spike amplitude calculation")
    
    ampMax = settings.spikeAmpMax
    ampMin = settings.spikeAmpMin
    amp = np.random.uniform(ampMin, ampMax)
    return amp

def calcScaledSpikeAmp(cell: Cell, site: Site, settings: Settings) -> list[float]:
    """スパイク振幅をスケーリングする"""
    spikeAmpList = cell.spikeAmpList
    d = calcDistance(cell, site)
    
    if settings.attenTime is None:
        raise ValueError("attenTime is required for spike amplitude scaling")
    
    # 距離減衰の計算（デバッグ用にログを追加）
    attenuation_factor = (d / settings.attenTime + 1)**2
    scaledSpikeAmpList = [amp / attenuation_factor for amp in spikeAmpList]
    
    return scaledSpikeAmpList

def calcDistance(cell: Cell, site: Site) -> float:
    """セルとサイトの距離を計算する"""
    return np.sqrt((cell.x - site.x) ** 2 + (cell.y - site.y) ** 2 + (cell.z - site.z) ** 2)

def simulateSpikeTemplate(settings: Settings) -> list[float]:
    """スパイクテンプレートをシミュレートする"""
    if settings.spikeType == "gabor":
        if (settings.gaborSigmaList is None or settings.gaborf0List is None or 
            settings.gaborthetaList is None or settings.spikeWidth is None):
            raise ValueError("gaborSigmaList, gaborf0List, gaborthetaList, and spikeWidth are required for gabor spike template")
        
        gaborSigmaList = np.random.choice(settings.gaborSigmaList)
        gaborf0List = np.random.choice(settings.gaborf0List)
        gaborthetaList = np.random.choice(settings.gaborthetaList)
        spikeTemplate = gabor(gaborSigmaList, gaborf0List, gaborthetaList, settings.fs, settings.spikeWidth)
        
        return spikeTemplate
    else:
        raise ValueError(f"spikeType '{settings.spikeType}' is not supported for template simulation")

def gabor(sigma: float, f0: float, theta: float, fs: float, spikeWidth: float) -> np.ndarray:
    """ガボール関数を生成する"""
    # spikeWidthはmsec単位、fsはHz単位
    # 時間軸を正しく設定（秒単位）
    x = np.linspace(-spikeWidth / 2, spikeWidth / 2, int(spikeWidth * fs / 1000))
    x = x / 1000  # msecを秒に変換
    
    # sigmaもmsec単位なので秒に変換
    sigma_sec = sigma / 1000
    
    y = np.exp(-x**2 / (2 * sigma_sec**2)) * np.cos(2 * np.pi * f0 * x + theta)
    y = y / np.max(np.abs(y))
    return y

def getFilteredSignal(signal: np.ndarray, fs: float, lowCutoffFreq: float, highCutoffFreq: float) -> np.ndarray:
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

def simulateDrift(settings: Settings, driftType: str = "linear") -> np.ndarray:
    """
    ドリフト信号をシミュレートする
    
    Args:
        settings: シミュレーション設定
        driftType: ドリフトタイプ ("linear", "exponential", "oscillatory", "random_walk")
    
    Returns:
        np.ndarray: ドリフト信号
    """
    duration = settings.duration
    fs = settings.fs
    signal_length = int(duration * fs)
    
    # ドリフトの振幅設定（デフォルト値）
    drift_amplitude = getattr(settings, 'drift_amplitude', 50.0)  # μV
    drift_frequency = getattr(settings, 'drift_frequency', 0.1)   # Hz
    
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

def addDriftToSignal(signal: np.ndarray, settings: Settings, driftType: str = "linear") -> tuple[np.ndarray, np.ndarray]:
    """
    信号にドリフトを追加する
    
    Args:
        signal: 元の信号
        settings: シミュレーション設定
        driftType: ドリフトタイプ
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (ドリフトが追加された信号, ドリフト信号)
    """
    if not hasattr(settings, 'enable_drift') or not settings.enable_drift:
        if isinstance(signal, list):
            signal = np.array(signal, dtype=np.float64)
        return signal, np.zeros_like(signal)
    
    drift = simulateDrift(settings, driftType)
    
    # 信号をnumpy配列に変換
    if isinstance(signal, list):
        signal = np.array(signal, dtype=np.float64)
    
    # ドリフトを追加
    signal_with_drift = signal + drift
    
    return signal_with_drift, drift

def generateCommonDrift(settings: Settings, num_sites: int) -> list[np.ndarray]:
    """
    電極全体で共通したドリフトを生成する
    
    Args:
        settings: シミュレーション設定
        num_sites: サイト数
    
    Returns:
        list[np.ndarray]: 各サイト用のドリフト信号（同じドリフトを複製）
    """
    if not hasattr(settings, 'enable_drift') or not settings.enable_drift:
        duration = settings.duration
        fs = settings.fs
        signal_length = int(duration * fs)
        return [np.zeros(signal_length) for _ in range(num_sites)]
    
    # 共通ドリフトを生成
    common_drift = simulateDrift(settings, settings.drift_type)
    
    # 各サイト用に複製
    drifts = [common_drift.copy() for _ in range(num_sites)]
    
    return drifts

def addCommonDriftToSites(sites: list[Site], settings: Settings) -> None:
    """
    すべてのサイトに共通ドリフトを追加する
    
    Args:
        sites: サイトのリスト
        settings: シミュレーション設定
    """
    if not hasattr(settings, 'enable_drift') or not settings.enable_drift:
        return
    
    # 共通ドリフトを生成
    drifts = generateCommonDrift(settings, len(sites))
    
    # 各サイトにドリフトを追加
    for i, (site, drift) in enumerate(zip(sites, drifts)):
        # signalRawにドリフトを追加
        if isinstance(site.signalRaw, list):
            site_signal = np.array(site.signalRaw, dtype=np.float64)
        else:
            site_signal = np.array(site.signalRaw, dtype=np.float64)
        
        site.signalRaw = (site_signal + drift).tolist()
        site.signalDrift = drift.tolist()


def simulatePowerLineNoise(settings: Settings, powerLineFreq: float = 50.0) -> np.ndarray:
    """
    電源ノイズ（50Hz/60Hz）をシミュレートする
    
    Args:
        settings: シミュレーション設定
        powerLineFreq: 電源周波数（Hz、デフォルト50Hz）
    
    Returns:
        np.ndarray: 電源ノイズ信号
    """
    duration = settings.duration
    fs = settings.fs
    signal_length = int(duration * fs)
    
    # 電源ノイズの振幅設定（デフォルト値）
    power_noise_amplitude = getattr(settings, 'power_noise_amplitude', 20.0)  # μV
    
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

def addPowerLineNoiseToSignal(signal: np.ndarray, settings: Settings, powerLineFreq: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """
    信号に電源ノイズを追加する
    
    Args:
        signal: 元の信号
        settings: シミュレーション設定
        powerLineFreq: 電源周波数
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (電源ノイズが追加された信号, 電源ノイズ信号)
    """
    if not hasattr(settings, 'enable_power_noise') or not settings.enable_power_noise:
        if isinstance(signal, list):
            signal = np.array(signal, dtype=np.float64)
        return signal, np.zeros_like(signal)
    
    power_noise = simulatePowerLineNoise(settings, powerLineFreq)
    
    # 信号をnumpy配列に変換
    if isinstance(signal, list):
        signal = np.array(signal, dtype=np.float64)
    
    # 電源ノイズを追加
    signal_with_power_noise = signal + power_noise
    
    return signal_with_power_noise, power_noise

def generateCommonPowerNoise(settings: Settings, num_sites: int, powerLineFreq: float = 50.0) -> list[np.ndarray]:
    """
    電極全体で共通した電源ノイズを生成する
    
    Args:
        settings: シミュレーション設定
        num_sites: サイト数
        powerLineFreq: 電源周波数
    
    Returns:
        list[np.ndarray]: 各サイト用の電源ノイズ信号（同じノイズを複製）
    """
    if not hasattr(settings, 'enable_power_noise') or not settings.enable_power_noise:
        duration = settings.duration
        fs = settings.fs
        signal_length = int(duration * fs)
        return [np.zeros(signal_length) for _ in range(num_sites)]
    
    # 共通電源ノイズを生成
    common_power_noise = simulatePowerLineNoise(settings, powerLineFreq)
    
    # 各サイト用に複製
    power_noises = [common_power_noise.copy() for _ in range(num_sites)]
    
    return power_noises

def addCommonPowerNoiseToSites(sites: list[Site], settings: Settings, powerLineFreq: float = 50.0) -> None:
    """
    すべてのサイトに共通電源ノイズを追加する
    
    Args:
        sites: サイトのリスト
        settings: シミュレーション設定
        powerLineFreq: 電源周波数
    """
    if not hasattr(settings, 'enable_power_noise') or not settings.enable_power_noise:
        return
    
    # 共通電源ノイズを生成
    power_noises = generateCommonPowerNoise(settings, len(sites), powerLineFreq)
    
    # 各サイトに電源ノイズを追加
    for i, (site, power_noise) in enumerate(zip(sites, power_noises)):
        # signalRawに電源ノイズを追加
        if isinstance(site.signalRaw, list):
            site_signal = np.array(site.signalRaw, dtype=np.float64)
        else:
            site_signal = np.array(site.signalRaw, dtype=np.float64)
        
        site.signalRaw = (site_signal + power_noise).tolist()
        site.signalPowerNoise = power_noise.tolist()


def calculateISI(spike_times: list[int], fs: float) -> list[float]:
    """
    スパイク時間からISI（Inter-Spike Interval）を計算する
    
    Args:
        spike_times: スパイク時間のリスト（サンプル単位）
        fs: サンプリング周波数（Hz）
    
    Returns:
        list[float]: ISIのリスト（秒単位）
    """
    if len(spike_times) < 2:
        return []
    
    # サンプル単位のISIを計算
    isi_samples = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
    
    # 秒単位に変換
    isi_seconds = [isi / fs for isi in isi_samples]
    
    return isi_seconds

def plotISI(spike_times: list[int], fs: float, cell_id: int = 0, save_path: str = None):
    """
    ISIの分布をプロットする
    
    Args:
        spike_times: スパイク時間のリスト
        fs: サンプリング周波数
        cell_id: 細胞ID（表示用）
        save_path: 保存パス（オプション）
    """
    import matplotlib.pyplot as plt
    
    isi_seconds = calculateISI(spike_times, fs)
    
    if not isi_seconds:
        print(f"細胞{cell_id}: ISIを計算できません（スパイク数が不足）")
        return
    
    # ミリ秒単位に変換
    isi_ms = [isi * 1000 for isi in isi_seconds]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ISIの時系列プロット
    ax1.plot(isi_ms, 'b-', alpha=0.7)
    ax1.set_xlabel('Spike Number')
    ax1.set_ylabel('ISI (ms)')
    ax1.set_title(f'ISI Time Series - Cell {cell_id}')
    ax1.grid(True, alpha=0.3)
    
    # ISIのヒストグラム
    ax2.hist(isi_ms, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('ISI (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'ISI Distribution - Cell {cell_id}')
    ax2.grid(True, alpha=0.3)
    
    # 統計情報を表示
    mean_isi = np.mean(isi_ms)
    std_isi = np.std(isi_ms)
    min_isi = np.min(isi_ms)
    max_isi = np.max(isi_ms)
    
    stats_text = f'Mean: {mean_isi:.2f} ms\nStd: {std_isi:.2f} ms\nMin: {min_isi:.2f} ms\nMax: {max_isi:.2f} ms'
    ax2.text(0.7, 0.9, stats_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ISIプロットを保存しました: {save_path}")
    
    plt.show()
    
    pass

def plotMultipleCellISI(cells: list, fs: float, save_path: str = None):
    """
    複数の細胞のISIを比較プロットする
    
    Args:
        cells: 細胞のリスト
        fs: サンプリング周波数
        save_path: 保存パス（オプション）
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, cell in enumerate(cells[:4]):  # 最初の4つの細胞のみ表示
        if not cell.spikeTimeList:
            continue
            
        isi_seconds = calculateISI(cell.spikeTimeList, fs)
        if not isi_seconds:
            continue
            
        isi_ms = [isi * 1000 for isi in isi_seconds]
        
        # ISIのヒストグラム
        axes[i].hist(isi_ms, bins=15, alpha=0.7, color=f'C{i}', edgecolor='black')
        axes[i].set_xlabel('ISI (ms)')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Cell {cell.id} ISI Distribution')
        axes[i].grid(True, alpha=0.3)
        
        # 統計情報
        mean_isi = np.mean(isi_ms)
        axes[i].axvline(mean_isi, color='red', linestyle='--', alpha=0.8, 
                       label=f'Mean: {mean_isi:.1f}ms')
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"複数細胞ISIプロットを保存しました: {save_path}")
    
    plt.show()

def plotRefractoryEffect(spike_times: list[int], fs: float, refractory_period: float, cell_id: int = 0):
    """
    不応期の効果を可視化する
    
    Args:
        spike_times: スパイク時間のリスト
        fs: サンプリング周波数
        refractory_period: 不応期（ms）
        cell_id: 細胞ID
    """
    import matplotlib.pyplot as plt
    
    if len(spike_times) < 2:
        print(f"細胞{cell_id}: スパイク数が不足しています")
        return
    
    isi_seconds = calculateISI(spike_times, fs)
    isi_ms = [isi * 1000 for isi in isi_seconds]
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ISIの時系列（不応期ライン付き）
    ax1.plot(isi_ms, 'b-', alpha=0.7, label='ISI')
    ax1.axhline(refractory_period, color='red', linestyle='--', alpha=0.8, 
               label=f'Refractory Period ({refractory_period}ms)')
    ax1.set_xlabel('Spike Number')
    ax1.set_ylabel('ISI (ms)')
    ax1.set_title(f'ISI vs Refractory Period - Cell {cell_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ISIのヒストグラム（不応期ライン付き）
    ax2.hist(isi_ms, bins=20, alpha=0.7, color='blue', edgecolor='black', label='ISI Distribution')
    ax2.axvline(refractory_period, color='red', linestyle='--', alpha=0.8, 
               label=f'Refractory Period ({refractory_period}ms)')
    ax2.set_xlabel('ISI (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'ISI Distribution vs Refractory Period - Cell {cell_id}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 不応期違反の統計
    violations = sum(1 for isi in isi_ms if isi < refractory_period)
    total_spikes = len(isi_ms)
    violation_rate = violations / total_spikes * 100 if total_spikes > 0 else 0
    
    stats_text = f'Total ISIs: {total_spikes}\nViolations: {violations}\nViolation Rate: {violation_rate:.1f}%'
    ax2.text(0.7, 0.9, stats_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    pass

