from pathlib import Path
import logging
import numpy as np
from scipy.signal import butter, filtfilt

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

def simulateSpikeTimes(settings: Settings) -> list[int]:
    """セルのスパイク時間をシミュレートする"""
    duration = settings.duration
    fs = settings.fs
    avgSpikeRate = settings.avgSpikeRate

    if settings.isRefractory:
        refractoryPeriod = settings.refractoryPeriod
    else:
        refractoryPeriod = 0

    isi = np.random.exponential(1 / avgSpikeRate, size=1000) + refractoryPeriod / 1000
    isi = np.ceil(isi * fs)
    spikeTimes = np.cumsum(isi)
    spikeTimes = spikeTimes[spikeTimes < int(duration * fs)]

    return spikeTimes

def simulateRecordingNoise(settings: Settings, noiseType: str) -> list[float]:
    """録音ノイズをシミュレートする"""
    duration = settings.duration
    fs = settings.fs
    noiseAmp = settings.noiseAmp

    if noiseType == "normal":
        noise = np.random.default_rng().integers(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
    elif noiseType == "gaussian":
        noise = np.random.normal(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
    else:
        raise ValueError(f"Invalid noise type: {noiseType}")

    return noise

def getRecordingNoiseFromTruth(settings: Settings) -> list[float]:
    """真の録音ノイズを取得する"""
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
    site.signalRaw = signal
    return signal
    
def calcSpikeAmp(settings: Settings) -> list[float]:
    """スパイク振幅を計算する"""
    ampMax = settings.spikeAmpMax
    ampMin = settings.spikeAmpMin
    amp = np.random.uniform(ampMin, ampMax)
    return amp

def calcScaledSpikeAmp(cell: Cell, site: Site, settings: Settings) -> list[float]:
    """スパイク振幅をスケーリングする"""
    spikeAmpList = cell.spikeAmpList
    d = calcDistance(cell, site)
    
    # 距離減衰の計算（デバッグ用にログを追加）
    attenuation_factor = (d / settings.attenTime + 1)**2
    scaledSpikeAmpList = [amp / attenuation_factor for amp in spikeAmpList]
    
    # デバッグ情報
    if len(spikeAmpList) > 0:
        logging.debug(f"距離={d:.2f}μm, 減衰係数={attenuation_factor:.2f}, 元振幅={spikeAmpList[0]:.2f}, 減衰後={scaledSpikeAmpList[0]:.2f}")
    
    return scaledSpikeAmpList

def calcDistance(cell: Cell, site: Site) -> float:
    """セルとサイトの距離を計算する"""
    return np.sqrt((cell.x - site.x) ** 2 + (cell.y - site.y) ** 2 + (cell.z - site.z) ** 2)

def simulateSpikeTemplate(settings: Settings) -> list[np.ndarray]:
    """スパイクテンプレートをシミュレートする"""
    gaborSigmaList = np.random.choice(settings.gaborSigmaList)
    gaborf0List = np.random.choice(settings.gaborf0List)
    gaborthetaList = np.random.choice(settings.gaborthetaList)
    spikeTemplate = gabor(gaborSigmaList, gaborf0List, gaborthetaList, settings.fs, settings.spikeWidth)
    
    return spikeTemplate

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
    b, a = butter(2, [lowCutoffFreq / (fs / 2), highCutoffFreq / (fs / 2)], btype='bandpass')
    filteredSignal = filtfilt(b, a, signal)
    return filteredSignal
