import numpy as np

def calculate_scaled_spike_amplitude(
    spikeAmpList: list[float],
    distance: float,
    attenTime: float
) -> list[float]:
    """スパイク振幅をスケーリングする"""

    attenuation_factor = (distance / attenTime + 1)**2
    return [amp / attenuation_factor for amp in spikeAmpList]


def calculate_distance_two_objects(target1, target2) -> float:
    """2つのオブジェクトの距離を計算する"""
    try:
        vx = target1.x - target2.x
        vy = target1.y - target2.y
        zv = target1.z - target2.z
        return np.sqrt(vx ** 2 + vy ** 2 + zv ** 2)
    except AttributeError:
        raise ValueError("target1 or target2 has no x, y, or z attributes")


def calculate_cosine_similarity(
    template1: list[float],
    template2: list[float]
) -> float:
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
