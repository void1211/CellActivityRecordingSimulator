import numpy as np
import logging
from tqdm import tqdm
from .Cell import Cell
from .Site import Site
from .calculate import calculate_spike_max_amplitude, calculate_scaled_spike_amplitude, calculate_distance_two_objects

from .Template import GaborTemplate, ExponentialTemplate

def make_noise_cells(
    duration: float, 
    fs: float, 
    sites: list[Site], 
    spikeSettings: dict,
    noiseSettings: dict
    ) -> list[Cell]:
    """
    3次元空間上に背景活動を生成する細胞を配置する
    
    Args:
        duration: シミュレーション時間
        fs: サンプリング周波数
        sites: 記録サイトのリスト
        margin: 記録サイトの周囲のマージン
        density: 背景活動細胞の密度
        inviolableArea: 禁止エリアの半径
        template_parameters: スパイクテンプレートのパラメータ
    Returns:
        list[Cell]: 生成された背景活動細胞のリスト
    """
    spikeType = spikeSettings["spikeType"]
    spikeAmpMax = spikeSettings["amplitudeMax"]
    spikeAmpMin = spikeSettings["amplitudeMin"]
    rate = spikeSettings["rate"]
    refractoryPeriod = spikeSettings["refractoryPeriod"]
    modelSettings = noiseSettings["model"]
    # 記録サイトの範囲を計算
    site_x_coords = [site.x for site in sites]
    site_y_coords = [site.y for site in sites]
    site_z_coords = [site.z for site in sites]
    
    min_x, max_x = min(site_x_coords), max(site_x_coords)
    min_y, max_y = min(site_y_coords), max(site_y_coords)
    min_z, max_z = min(site_z_coords), max(site_z_coords)
    
    # マージンを追加して配置範囲を拡張
    volume_x = (max_x - min_x) + 2 * modelSettings["margin"]
    volume_y = (max_y - min_y) + 2 * modelSettings["margin"]
    volume_z = (max_z - min_z) + 2 * modelSettings["margin"]
    
    # 体積をmm³に変換（μm³ → mm³）
    volume_mm3 = (volume_x * volume_y * volume_z) / (1000**3)
    
    # 細胞数を計算（密度 × 体積）
    cell_count = int(modelSettings["density"] * volume_mm3)
    
    # 細胞をランダムに配置
    noise_cells = []
    attempts = 0
    max_attempts = 1000  # 無限ループ防止
    
    logging.info(f"Generating {cell_count} noise cells...")
    for i in tqdm(range(cell_count)):
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            
            # ランダムな位置を生成
            x = np.random.uniform(min_x - modelSettings["margin"], max_x + modelSettings["margin"])
            y = np.random.uniform(min_y - modelSettings["margin"], max_y + modelSettings["margin"])
            z = np.random.uniform(min_z - modelSettings["margin"], max_z + modelSettings["margin"])
            
            # 禁止エリア外にあるかチェック（記録サイトの周囲inviolableAreaの距離内は禁止）
            is_in_violable_area = False
            for site in sites:
                distance = np.sqrt((x - site.x)**2 + (y - site.y)**2 + (z - site.z)**2)
                if distance <= modelSettings["inviolableArea"]:
                    is_in_violable_area = True
                    break
            if not is_in_violable_area:
                break  # 禁止エリア外ならループを抜ける
        
        if attempts >= max_attempts:
            logging.warning(f"細胞 {i} の配置に失敗しました。最大試行回数に達しました。")
            continue
        
        # 細胞を生成
        cell = Cell(
            id=i,
            x=x,
            y=y,
            z=z
        )
        
        # スパイク活動をシミュレート
        cell.spikeTimeList = make_spike_times(duration, fs, rate, refractoryPeriod)
        cell.spikeAmpList = [calculate_spike_max_amplitude(spikeAmpMax, spikeAmpMin) for _ in cell.spikeTimeList]
        if spikeType == "gabor":  
            cell.spikeTemp = GaborTemplate(fs, spikeSettings).generate()
        elif spikeType == "exponential":
            cell.spikeTemp = ExponentialTemplate(fs, spikeSettings).generate()
        else:
            raise ValueError(f"Invalid spikeType")
        noise_cells.append(cell)

    
    return noise_cells

def make_background_activity(duration: float, fs: float, noise_cells: list[Cell], site: Site, attenTime: float) -> list[float]:
    """
    ノイズ細胞の活動を記録サイトの信号に追加する
    
    Args:
        noise_cells: ノイズ細胞のリスト
        sites: 記録サイトのリスト
        attenTime: 減衰時間
    """
    if not noise_cells:
        logging.warning("ノイズ細胞が存在しません")
        return
    
    # サイトの信号をnumpy配列に変換
    site_signal = np.zeros(int(duration * fs))
    
    added_spikes = 0
    logging.info(f"Adding spikes to site {site.id}...")
    for cell_idx, cell in tqdm(enumerate(noise_cells), total=len(noise_cells)):
        # 各細胞からの信号を計算
        scaled_amps = calculate_scaled_spike_amplitude(cell.spikeAmpList, calculate_distance_two_objects(cell, site), attenTime)
        # スパイクを信号に追加
        spikeTimes = cell.spikeTimeList
        spikeTemp = cell.spikeTemp
        
        # ピーク位置を正しく計算（負のピークも考慮）
        if np.min(spikeTemp) < 0 and abs(np.min(spikeTemp)) > abs(np.max(spikeTemp)):
            # 負のピークが主成分の場合
            peak = np.argmin(spikeTemp)
            peak_type = "negative"
        else:
            # 正のピークが主成分の場合
            peak = np.argmax(spikeTemp)
            peak_type = "positive"
        
        # デバッグ情報（最初の5個の細胞のみ）
        if cell_idx < 5:
            logging.debug(f"  細胞{cell_idx}: ピークタイプ={peak_type}, ピーク位置={peak}, "
                         f"テンプレート範囲=[{np.min(spikeTemp):.2f}, {np.max(spikeTemp):.2f}]")
        
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
                distance = calculate_distance_two_objects(cell, site)
                logging.debug(f"  細胞{cell_idx}: 距離={distance:.2f}μm, 追加スパイク数={cell_added_spikes}")
    
    return site_signal.tolist()

def make_spike_times(duration:float, fs:float, rate:float, refractoryPeriod:float=2.0) -> list[int]:
    """セルのスパイク時間をシミュレートする"""

    # より現実的な不応期モデル
    if refractoryPeriod > 0:
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
        isi = np.random.exponential(1 / rate, size=10000)
        isi = np.ceil(isi * fs)
        spikeTimes = np.cumsum(isi)
        spikeTimes = spikeTimes[spikeTimes < int(duration * fs)]
        return spikeTimes



