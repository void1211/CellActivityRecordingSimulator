from pathlib import Path

import numpy as np

from Cell import Cell
from Site import Site
from Settings import Settings

def getProjectRoot() -> Path:
    """プロジェクトのルートディレクトリを取得する"""
    return Path(__file__).resolve().parents[2]

def simulateSpikeTimes(cell: Cell, settings: Settings) -> list[int]:
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

    if noiseType == "gaussian":
        noise = np.random.normal(0, noiseAmp, size=int(duration * fs))
    elif noiseType == "truth":
        noise = np.loadtxt(settings.pathTruthNoise)
    else:
        raise ValueError(f"Invalid noise type: {noiseType}")

    return noise

def getRecordingNoiseFromTruth(settings: Settings) -> list[float]:
    """真の録音ノイズを取得する"""
    noise = np.load(settings.pathTruthNoise)
    return noise

def addSpikeToSignal(cell: Cell, site: Site) -> list[float]:
    """スパイクを信号に追加する"""
    signal = site.signalRaw
    spikeTimes = cell.spikeTimeList
    spikeAmpList = cell.spikeAmpList
    spikeTemp = cell.spikeTemp
    peak = np.argmax(np.abs(spikeTemp))
    for spikeTime, spikeAmp in zip(spikeTimes, spikeAmpList):
        start = spikeTime - peak
        end = start + len(spikeTemp)
        if not (0 <= start and end <= len(signal)):
            continue
        signal[start:end] += spikeAmp * spikeTemp
    site.signalRaw = signal
    return signal
    
def calcScaledSpikeAmp(cell: Cell, site: Site, settings: Settings) -> list[float]:
    """スパイク振幅をスケーリングする"""
    spikeAmpList = cell.spikeAmpList
    d = calcDistance(cell, site)
    scaledSpikeAmpList = spikeAmpList / (d / settings.attenTime + 1)**2
    cell.spikeAmpList = scaledSpikeAmpList
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
    x = np.linspace(-spikeWidth / 2, spikeWidth / 2, int(spikeWidth * fs))
    y = np.exp(-x**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f0 * x + theta)
    y = y / np.max(np.abs(y))
    return y
