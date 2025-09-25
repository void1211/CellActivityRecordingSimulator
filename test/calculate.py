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

