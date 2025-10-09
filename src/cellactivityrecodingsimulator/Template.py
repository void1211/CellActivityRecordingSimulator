import numpy as np
from spikeinterface.core.generate import generate_single_fake_waveform
import logging
from typing import Dict, Type
from pathlib import Path
import json

from .calculate import calculate_cosine_similarity

# テンプレートクラスのマッピング（後で設定）
TEMPLATE_CLASSES: Dict[str, Type['BaseTemplate']] = {}
class BaseTemplate:
    def __init__(self):
        self.template = []

    def get_template(self) -> np.ndarray:
        """テンプレートを取得する"""
        return self.template

    def set_template(self, template: np.ndarray):
        """テンプレートを設定する"""
        self.template = template

    @staticmethod
    def _choose_value(settings: dict, parameter_name:str):
        param_value = settings.get(parameter_name)
        if param_value is None:
            raise ValueError(f"Parameter '{parameter_name}' is not provided")
        if len(param_value) == 1:
            return param_value[0]
        else:
            return BaseTemplate._choose_value_from_list(settings, parameter_name)

    @staticmethod
    def _choose_value_from_list(settings: dict, parameter_name:str):
        if settings["randType"] == "list":
            return np.random.choice(settings[parameter_name])
        elif settings["randType"] == "range":
            return np.random.uniform(settings[parameter_name][0], settings[parameter_name][1])
        else:
            raise ValueError(f"Invalid randType")

    @classmethod
    def generate(cls, settings) -> "GaborTemplate" or "ExponentialTemplate":
        """
        設定に基づいて適切なテンプレートクラスを選択してテンプレートを生成する
        
        Args:
            settings: シミュレーション設定（dictまたはSettingsオブジェクト）
        
        Returns:
            GaborTemplate or ExponentialTemplate: 生成されたテンプレート
        """
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        spike_type = settings["spikeSettings"]["spikeType"]
        return TEMPLATE_CLASSES[spike_type].generate(settings)

    def generate_similar_template(self, settings: dict) -> 'BaseTemplate':
        """基準テンプレートと類似度制御されたテンプレートを生成する"""
        sim_settings = settings["templateSimilarityControlSettings"]
        min_sim, max_sim = sim_settings["min_cosine_similarity"], sim_settings["max_cosine_similarity"]
        
        def is_in_range(similarity):
            return min_sim <= similarity <= max_sim
        
        def create_template_from_array(template_array):
            spike_type = settings["spikeSettings"]["spikeType"]
            return TEMPLATE_CLASSES[spike_type](template=template_array)

        for _ in range(sim_settings["similarity_control_attempts"]):
            new_template_obj = self.generate(settings)
            new_template = new_template_obj.get_template()
            
            # 通常のテンプレートをチェック
            if is_in_range(calculate_cosine_similarity(self.get_template(), new_template)):
                return new_template_obj
            
            # 反転テンプレートをチェック
            inverted_template = -1.0 * new_template
            if is_in_range(calculate_cosine_similarity(self.get_template(), inverted_template)):
                return create_template_from_array(inverted_template)
            
        
        # 最大試行回数に達した場合、最も類似度が中央値に近いテンプレートを返す
        logging.warning(f"類似度制御の最大試行回数（{sim_settings['similarity_control_attempts']}）に達しました。")
        
        target_similarity = (min_sim + max_sim) / 2
        best_template, best_distance = None, float('inf')
        
        for _ in range(10):
            new_template_obj = self.generate(settings)
            similarity = calculate_cosine_similarity(self.get_template(), new_template_obj.get_template())
            distance = abs(similarity - target_similarity)
            
            if distance < best_distance:
                best_distance, best_template = distance, new_template_obj
        
        return best_template

    @classmethod
    def load_spike_templates(cls, path: Path) -> list['BaseTemplate']:
        """ファイルからスパイクテンプレートを読み込む"""
        with open(path, "r") as f:
            data = json.load(f)
        
        return [
            BaseTemplate(template=np.array(data["spikeTemplate"][i])) 
            for i in range(len(data["id"]))
        ]

class GaborTemplate(BaseTemplate):
    def __init__(self):
        super().__init__()
        self.template = []

    @classmethod
    def generate(cls, settings: dict) -> "GaborTemplate": 
        gaborSettings = settings["spikeSettings"]["gabor"]
        fs = settings["baseSettings"]["fs"]
        width = gaborSettings["width"]
        sigma = BaseTemplate._choose_value(settings=gaborSettings, parameter_name="sigma")
        f0 = BaseTemplate._choose_value(settings=gaborSettings, parameter_name="f0")
        theta = BaseTemplate._choose_value(settings=gaborSettings, parameter_name="theta")
        x = np.linspace(-width / 2, width / 2, int(width * fs / 1000))
        x = x / 1000
        sigma_sec = sigma / 1000
        gabortheta_rad = theta * np.pi / 180

        template = np.exp(-x**2 / (2 * sigma_sec**2)) * np.cos(2 * np.pi * f0 * x + gabortheta_rad)
        template = template / np.max(np.abs(template))
        gaborTemplate = GaborTemplate()
        gaborTemplate.set_template(template)
        return gaborTemplate

class ExponentialTemplate(BaseTemplate):
    def __init__(self):
        super().__init__()
        self.template = []

    @classmethod
    def generate(cls, settings: dict) -> "ExponentialTemplate":
        exponentialSettings = settings["spikeSettings"]["exponential"]
        fs = settings["baseSettings"]["fs"]
        ms_before = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="ms_before")
        ms_after = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="ms_after")
        negative_amplitude = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="negative_amplitude")
        positive_amplitude = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="positive_amplitude")
        depolarization_ms = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="depolarization_ms")
        repolarization_ms = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="repolarization_ms")
        recovery_ms = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="recovery_ms")
        smooth_ms = BaseTemplate._choose_value(settings=exponentialSettings, parameter_name="smooth_ms")
        
        template = generate_single_fake_waveform(
        sampling_frequency=fs,
        ms_before=ms_before,
        ms_after=ms_after,
        negative_amplitude=negative_amplitude,
        positive_amplitude=positive_amplitude,
        depolarization_ms=depolarization_ms,
        repolarization_ms=repolarization_ms,
        recovery_ms=recovery_ms,
        smooth_ms=smooth_ms)    
        # ピークを絶対値１に調整
        template = template / np.max(np.abs(template))
        exponentialTemplate = ExponentialTemplate()
        exponentialTemplate.set_template(template)
        return exponentialTemplate

# テンプレートクラスを登録
TEMPLATE_CLASSES["gabor"] = GaborTemplate
TEMPLATE_CLASSES["exponential"] = ExponentialTemplate
