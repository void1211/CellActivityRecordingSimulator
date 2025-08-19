import numpy as np


def calcSpikeAmp(ampMax: float, ampMin: float) -> float:
    """スパイク振幅を計算する"""
    amp = np.random.uniform(ampMin, ampMax)
    return amp

def calcScaledSpikeAmp(spikeAmpList: list[float], distance: float, attenTime: float) -> list[float]:
    """スパイク振幅をスケーリングする"""

    attenuation_factor = (distance / attenTime + 1)**2
    scaledSpikeAmpList = [amp / attenuation_factor for amp in spikeAmpList]
    
    return scaledSpikeAmpList

def calcDistance(target1, target2) -> float:
    """2つのオブジェクトの距離を計算する"""
    if hasattr(target1, "x") and hasattr(target1, "y") and hasattr(target1, "z"):
        if hasattr(target2, "x") and hasattr(target2, "y") and hasattr(target2, "z"):
            return np.sqrt((target1.x - target2.x) ** 2 + (target1.y - target2.y) ** 2 + (target1.z - target2.z) ** 2)
        else:
            raise ValueError("target2 has no x, y, or z attributes")
    else:
        raise ValueError("target1 has no x, y, or z attributes")


def calculateCosineSimilarity(template1: list[float], template2: list[float]) -> float:
    """
    2つのスパイクテンプレート間のコサイン類似度を計算する
    
    Args:
        template1: 1つ目のテンプレート
        template2: 2つ目のテンプレート
    
    Returns:
        float: コサイン類似度(-1.0-1.0)
    """
    # リストをnumpy配列に変換
    t1 = np.array(template1)
    t2 = np.array(template2)
    
    # 正規化
    t1_norm = t1 / np.linalg.norm(t1)
    t2_norm = t2 / np.linalg.norm(t2)
    
    # コサイン類似度を計算
    similarity = np.dot(t1_norm, t2_norm)
    
    # 値を-1.0-1.0の範囲に制限
    return max(-1.0, min(1.0, similarity))

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