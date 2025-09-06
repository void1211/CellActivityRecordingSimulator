from pathlib import Path
import logging
import numpy as np
from scipy.signal import butter, filtfilt

def makeSaveDir(pathSaveDir: Path) -> Path:
    """保存ディレクトリを作成する"""
    pathSaveDir.mkdir(parents=True, exist_ok=True)
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
    return pathSaveDir    

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

