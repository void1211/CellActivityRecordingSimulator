import numpy as np
import logging

from Cell import Cell
from Site import Site
from Settings import Settings
from simulate import simulateSpikeTimes, simulateSpikeTemplate
from calculate import calcSpikeAmp, calculateCosineSimilarity


def generateNoiseCells(duration: float, fs: float, sites: list[Site], margin: float, density: float, 
spikeAmpMax: float = 100, spikeAmpMin: float = 90, spikeType: str = "gabor", randType: str = "range",
gaborSigma: list[float] = [0.2, 0.4], gaborf0: list[float] = [300, 500], gabortheta: list[float] = [0, 360],
spikeWidth: float = 4) -> list[Cell]:
    """
    3次元空間上に背景活動を生成する細胞を配置する
    
    Args:
        duration: シミュレーション時間
        fs: サンプリング周波数
        sites: 記録サイトのリスト
        margin: 記録サイトの周囲のマージン
        density: 背景活動細胞の密度
        spikeAmpMax: スパイク振幅の最大値
        spikeAmpMin: スパイク振幅の最小値
        spikeType: スパイクテンプレートのタイプ
        randType: ガボール関数のシグマ、f0、thetaの生成タイプ
        gaborSigma: ガボール関数のシグマ
        gaborf0: ガボール関数のf0
        gabortheta: ガボール関数のtheta
        spikeWidth: スパイク幅
    Returns:
        list[Cell]: 生成された背景活動細胞のリスト
    """
    
    # 記録サイトの範囲を計算
    site_x_coords = [site.x for site in sites]
    site_y_coords = [site.y for site in sites]
    site_z_coords = [site.z for site in sites]
    
    min_x, max_x = min(site_x_coords), max(site_x_coords)
    min_y, max_y = min(site_y_coords), max(site_y_coords)
    min_z, max_z = min(site_z_coords), max(site_z_coords)
    
    # マージンを追加して配置範囲を拡張
    volume_x = (max_x - min_x) + 2 * margin
    volume_y = (max_y - min_y) + 2 * margin
    volume_z = (max_z - min_z) + 2 * margin
    
    # 体積をmm³に変換（μm³ → mm³）
    volume_mm3 = (volume_x * volume_y * volume_z) / (1000**3)
    
    # 細胞数を計算（密度 × 体積）
    cell_count = int(density * volume_mm3)
    
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
        cell.spikeTimeList = simulateSpikeTimes(duration, fs)
        cell.spikeAmpList = [calcSpikeAmp(spikeAmpMax, spikeAmpMin) for _ in cell.spikeTimeList]
        cell.spikeTemp = simulateSpikeTemplate(fs, spikeType, randType, spikeWidth,
            gaborSigma=gaborSigma, gaborf0=gaborf0, gabortheta=gabortheta)
        
        noise_cells.append(cell)
    
    return noise_cells

def generate_similar_templates_for_group(cells: list[Cell], settings: Settings, group_id: int) -> list[list[float]]:
    """
    同じグループの細胞に対して類似度制御されたスパイクテンプレートを生成する
    
    Args:   
        cells: 細胞のリスト
        settings: シミュレーション設定
        group_id: 対象のグループID
    
    Returns:
        list[list[float]]: 生成されたテンプレートのリスト
    """
    # 指定されたグループの細胞を抽出
    group_cells = [cell for cell in cells if cell.group == group_id]
    
    if not group_cells:
        logging.warning(f"グループ {group_id} の細胞が見つかりません")
        return []
    
    templates = []
    
    for i, cell in enumerate(group_cells):
        if i == 0:
            # 最初の細胞は基準テンプレートを生成
            template = simulateSpikeTemplate(settings)
            templates.append(template)
            logging.info(f"グループ {group_id} の基準テンプレートを生成しました")
        else:
            # 2番目以降の細胞は類似度制御されたテンプレートを生成
            template = generate_similar_template(templates[0], settings)
            templates.append(template)
            similarity = calculateCosineSimilarity(templates[0], template)
            logging.info(f"グループ {group_id} の細胞 {cell.id} のテンプレートを生成しました（類似度: {similarity:.3f}）")
    
    return templates

def generate_similar_template(fs, spikeType, base_template: list[float], min_similarity: float, max_similarity: float, max_attempts: int) -> list[float]:
    """
    基準テンプレートと類似度制御されたテンプレートを生成する
    
    Args:
        base_template: 基準テンプレート
        settings: シミュレーション設定
    
    Returns:
        list[float]: 類似度制御されたテンプレート
    """
    
    for attempt in range(max_attempts):
        # 新しいテンプレートを生成（設定の制限を無視）
        new_template = simulateSpikeTemplate(fs, spikeType, "range", len(base_template),
            gaborSigmaMin=0.2, gaborf0Min=300, gaborthetaMin=0,
            gaborSigmaMax=0.4, gaborf0Max=500, gaborthetaMax=360)
        
        # 類似度を計算
        similarity = calculateCosineSimilarity(base_template, new_template)
        
        # 類似度が指定範囲内にあるかチェック
        if min_similarity <= similarity <= max_similarity:
            return new_template
    
    # 最大試行回数に達した場合は、最も近いテンプレートを返す
    logging.warning(f"類似度制御の最大試行回数（{max_attempts}）に達しました。最も近いテンプレートを使用します。")
    
    best_template = None
    best_similarity = -1.0
    
    for _ in range(10):  # 最後の10回の試行
        template = simulateSpikeTemplate(fs, spikeType, "range", len(base_template),
            gaborSigmaMin=0.2, gaborf0Min=300, gaborthetaMin=0,
            gaborSigmaMax=0.4, gaborf0Max=500, gaborthetaMax=360)
        similarity = calculateCosineSimilarity(base_template, template)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_template = template
    
    return best_template
