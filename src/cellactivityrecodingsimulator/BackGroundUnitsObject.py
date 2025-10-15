from .Unit import Unit
from .ProbeObject import ProbeObject
from .Contact import Contact
from .calculate import calculate_scaled_spike_amplitude, calculate_distance_two_objects

import numpy as np
import logging
from tqdm import tqdm
import time

class BGUnitsObject:
    def __init__(self):
        self.units = []

    def __str__(self):
        return f"BGUnitsObject - {len(self.units)} units"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def generate(cls, settings, probe: ProbeObject) -> "BGUnitsObject":
        """3次元空間上に背景活動を生成する細胞を配置する"""
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        bg_units = cls()
        modelSettings = settings["noiseSettings"]["model"]
        contacts = probe.contacts
        # 記録サイトの範囲を計算
        contact_x_coords = [contact.x for contact in contacts]
        contact_y_coords = [contact.y for contact in contacts]
        contact_z_coords = [contact.z for contact in contacts]
        
        min_x, max_x = min(contact_x_coords), max(contact_x_coords)
        min_y, max_y = min(contact_y_coords), max(contact_y_coords)
        min_z, max_z = min(contact_z_coords), max(contact_z_coords)
        
        # マージンを追加して配置範囲を拡張
        volume_x = (max_x - min_x) + 2 * modelSettings["margin"]
        volume_y = (max_y - min_y) + 2 * modelSettings["margin"]
        volume_z = (max_z - min_z) + 2 * modelSettings["margin"]
        
        # 体積をmm³に変換（μm³ → mm³）
        volume_mm3 = (volume_x * volume_y * volume_z) / (1000**3)
        
        # 細胞数を計算（密度 × 体積）
        unit_count = int(modelSettings["density"] * volume_mm3)
        
        # 細胞をランダムに配置
        bg_units_list = []
        attempts = 0
        max_attempts = 1000  # 無限ループ防止
        
        logging.info(f"=== ノイズ細胞生成 ===")
        for i in tqdm(range(unit_count), desc="背景活動細胞生成中", total=unit_count):
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                
                # ランダムな位置を生成
                x = np.random.uniform(min_x - modelSettings["margin"], max_x + modelSettings["margin"])
                y = np.random.uniform(min_y - modelSettings["margin"], max_y + modelSettings["margin"])
                z = np.random.uniform(min_z - modelSettings["margin"], max_z + modelSettings["margin"])
                
                # 禁止エリア外にあるかチェック（記録サイトの周囲inviolableAreaの距離内は禁止）
                is_in_violable_area = False
                for contact in contacts:
                    distance = np.sqrt((x - contact.x)**2 + (y - contact.y)**2 + (z - contact.z)**2)
                    if distance <= modelSettings["inviolableArea"]:
                        is_in_violable_area = True
                        break
                if not is_in_violable_area:
                    break  # 禁止エリア外ならループを抜ける
            
            if attempts >= max_attempts:
                logging.warning(f"細胞 {i} の配置に失敗しました。最大試行回数に達しました。")
                continue
            
            # 細胞を生成
            unit = Unit.generate(
                group=0,
                id=i,
                x=x,
                y=y,
                z=z
            )
            unit.set_templateObject(settings=settings)
            unit.set_spike_time(settings=settings)
            unit.set_amplitudes(settings=settings)
            bg_units_list.append(unit)

        bg_units.units = bg_units_list
        return bg_units

    def make_background_activity(self, contact: Contact, attenTime: float, settings) -> Contact:
        """ノイズ細胞の活動を記録サイトの信号に追加する"""
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        # 信号を初期化
        duration_samples = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
        contact_signal = np.zeros(duration_samples)
        
        added_spikes = 0
        logging.info(f"=== ノイズ細胞活動追加 ===")
        for unit_idx, unit in tqdm(enumerate(self.units), desc="ノイズ細胞活動追加中", total=len(self.units), position=1):
            # 各細胞からの信号を計算
            distance = calculate_distance_two_objects(unit, contact)
            scaled_amps = calculate_scaled_spike_amplitude(
                unit.spikeAmpList, 
                distance, 
                attenTime
                )
            
            template = unit.templateObject.get_template() 
            
            # テンプレートが空または無効な場合はスキップ
            if len(template) == 0:
                continue
            
            # ピーク位置を正しく計算（負のピークも考慮）
            if np.min(template) < 0 and abs(np.min(template)) > abs(np.max(template)):
                # 負のピークが主成分の場合
                peak = np.argmin(template)
                peak_type = "negative"
            else:
                # 正のピークが主成分の場合
                peak = np.argmax(template)
                peak_type = "positive"
            
            # デバッグ情報（最初の5個の細胞のみ）
            if unit_idx < 5:
                logging.debug(f"  細胞{unit_idx}: ピークタイプ={peak_type}, ピーク位置={peak}, "
                            f"テンプレート範囲=[{np.min(template):.2f}, {np.max(template):.2f}]")
            
            unit_added_spikes = 0
            for spikeTime, spikeAmp in zip(unit.spikeTimeList, scaled_amps):
                start = int(spikeTime - peak)
                end = int(start + len(template))
                if not (0 <= start and end <= len(contact_signal)):
                    continue
                contact_signal[start:end] += spikeAmp * template
                unit_added_spikes += 1
            
            if unit_added_spikes > 0:
                added_spikes += unit_added_spikes
                if unit_idx < 5:  # 最初の5個の細胞のみログ出力
                    distance = calculate_distance_two_objects(unit, contact)
                    logging.debug(f"  細胞{unit_idx}: 距離={distance:.2f}μm, 追加スパイク数={unit_added_spikes}")
            
            time.sleep(0.1)
        
        contact.set_signal("background", contact_signal)
        return contact

    def to_dict(self):
        return {
            "units": [unit.to_dict() for unit in self.units]
        }

    @classmethod
    def from_dict(cls, data: dict):
        units = [Unit.from_dict(unit) for unit in data["units"]]
        bg_units = cls()
        bg_units.units = units
        return bg_units
