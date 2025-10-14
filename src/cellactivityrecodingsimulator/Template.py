import numpy as np
from spikeinterface.core.generate import generate_single_fake_waveform
import logging

from .calculate import calculate_cosine_similarity

class BaseTemplate:
    def __init__(self, spikeSettings: dict):
        self._template = None
        self._spikeSettings = spikeSettings
    
    @property
    def spikeSettings(self) -> dict:
        return self._spikeSettings
    
    @spikeSettings.setter
    def spikeSettings(self, spikeSettings: dict):
        self._spikeSettings = spikeSettings

    @property
    def template(self) -> np.ndarray:
        return self._template
    
    @template.setter
    def template(self, template: np.ndarray):
        self._template = template

    def _choose_value(self, settings: dict, parameter_name:str):
        param_value = settings.get(parameter_name)
        if param_value is None:
            raise ValueError(f"Parameter '{parameter_name}' is not provided")
        if len(param_value) == 1:
            return param_value[0]
        else:
            return self._choose_value_from_list(settings, parameter_name)

    def _choose_value_from_list(self, settings: dict, parameter_name:str):
        if settings["randType"] == "list":
            return np.random.choice(settings[parameter_name])
        elif settings["randType"] == "range":
            return np.random.uniform(settings[parameter_name][0], settings[parameter_name][1])
        else:
            raise ValueError(f"Invalid randType")

class GaborTemplate(BaseTemplate):
    def __init__(self, fs: float, spikeSettings: dict):
        super().__init__(spikeSettings)
        gaborSettings = spikeSettings["gabor"]
        self._fs = fs
        self._width = gaborSettings["width"]
        self._sigma = self._choose_value(gaborSettings, "sigma")
        self._f0 = self._choose_value(gaborSettings, "f0")
        self._theta = self._choose_value(gaborSettings, "theta")

    def generate(self) -> np.ndarray: 
        self.template = self._gabor()
        return self.template
    
    def _gabor(self) -> np.ndarray:
        """ガボール関数を生成する"""
        x = np.linspace(-self._width / 2, self._width / 2, int(self._width * self._fs / 1000))
        x = x / 1000
        sigma_sec = self._sigma / 1000
        gabortheta_rad = self._theta * np.pi / 180
        
        y = np.exp(-x**2 / (2 * sigma_sec**2)) * np.cos(2 * np.pi * self._f0 * x + gabortheta_rad)
        y = y / np.max(np.abs(y))
        return y

class ExponentialTemplate(BaseTemplate):
    def __init__(self, fs: float, spikeSettings: dict):
        super().__init__(spikeSettings)
        exponentialSettings = spikeSettings["exponential"]
        self._fs = fs
        self._ms_before = self._choose_value(exponentialSettings, "ms_before")
        self._ms_after = self._choose_value(exponentialSettings, "ms_after")
        self._negative_amplitude = self._choose_value(exponentialSettings, "negative_amplitude")
        self._positive_amplitude = self._choose_value(exponentialSettings, "positive_amplitude")
        self._depolarization_ms = self._choose_value(exponentialSettings, "depolarization_ms")
        self._repolarization_ms = self._choose_value(exponentialSettings, "repolarization_ms")
        self._recovery_ms = self._choose_value(exponentialSettings, "recovery_ms")
        self._smooth_ms = self._choose_value(exponentialSettings, "smooth_ms")

    def generate(self) -> np.ndarray:
        template = generate_single_fake_waveform(
        sampling_frequency=self._fs,
        ms_before=self._ms_before,
        ms_after=self._ms_after,
        negative_amplitude=self._negative_amplitude,
        positive_amplitude=self._positive_amplitude,
        depolarization_ms=self._depolarization_ms,
        repolarization_ms=self._repolarization_ms,
        recovery_ms=self._recovery_ms,
        smooth_ms=self._smooth_ms)    
        # ピークを絶対値１に調整
        self.template = template / np.max(np.abs(template))
        return self.template
    

def make_similar_templates(fs: float, 
    templateSimilarityControlSettings: dict, 
    spikeSettings: dict, 
    num_cells: int
    ) -> list[list[float]]:
    """
    同じグループの細胞に対して類似度制御されたスパイクテンプレートを生成する
    
    Args:   
        cells: 細胞のリスト
        settings: シミュレーション設定
        group_id: 対象のグループID
    
    Returns:
        list[list[float]]: 生成されたテンプレートのリスト
    """

    
    templates = []

    
    for i in range(num_cells):
        if i == 0:
            # 最初の細胞は基準テンプレートを生成
            if spikeSettings["spikeType"] == "gabor":
                template = GaborTemplate(fs, spikeSettings).generate()
            elif spikeSettings["spikeType"] == "exponential":
                template = ExponentialTemplate(fs, spikeSettings).generate()
            else:
                raise ValueError(f"Invalid spikeType")
            templates.append(template)
            logging.info(f"基準テンプレートを生成しました")
        else:
            # 2番目以降の細胞は類似度制御されたテンプレートを生成
            template = make_similar_template(
                fs, 
                templates[0], 
                templateSimilarityControlSettings, 
                spikeSettings)
            templates.append(template)
            similarity = calculate_cosine_similarity(templates[0], template)
            logging.info(f"{i}番目のテンプレートを生成しました（類似度: {similarity:.3f})")
    
    return templates

def make_similar_template(fs, 
    base_template: list[float], 
    templateSimilarityControlSettings: dict, 
    spikeSettings: dict
    ) -> list[float]:
    """
    基準テンプレートと類似度制御されたテンプレートを生成する
    
    Args:
        base_template: 基準テンプレート
        settings: シミュレーション設定
    
    Returns:
        list[float]: 類似度制御されたテンプレート
    """
    
    for attempt in range(templateSimilarityControlSettings["similarity_control_attempts"]):
        # 新しいテンプレートを生成（設定の制限を無視）
        if spikeSettings["spikeType"] == "gabor":
            new_template = GaborTemplate(fs, spikeSettings).generate()
        elif spikeSettings["spikeType"] == "exponential":
            new_template = ExponentialTemplate(fs, spikeSettings).generate()
        else:
            raise ValueError(f"Invalid spikeType")
        
        # 類似度を計算
        similarity = calculate_cosine_similarity(base_template, new_template)
        
        # 類似度が指定範囲内にあるかチェック
        if templateSimilarityControlSettings["min_cosine_similarity"] <= similarity <= templateSimilarityControlSettings["max_cosine_similarity"]:
            return new_template
        
        # 反転させたテンプレートもチェック
        inverted_template = -1.0 * new_template
        inverted_similarity = calculate_cosine_similarity(base_template, inverted_template)
        
        # 反転させたテンプレートが指定範囲内にあるかチェック
        if templateSimilarityControlSettings["min_cosine_similarity"] <= inverted_similarity <= templateSimilarityControlSettings["max_cosine_similarity"]:
            return inverted_template
        
    
    # 最大試行回数に達した場合は、最も近いテンプレートを返す
    max_attempts = templateSimilarityControlSettings["similarity_control_attempts"]
    logging.warning(f"類似度制御の最大試行回数（{max_attempts}）に達しました。最も範囲に近いテンプレートを使用します。")
    
    best_template = None
    best_similarity = -1.0
    
    for _ in range(10):  # 最後の10回の試行
        if spikeSettings["spikeType"] == "gabor":
            new_template = GaborTemplate(fs, spikeSettings).generate()
        elif spikeSettings["spikeType"] == "exponential":
            new_template = ExponentialTemplate(fs, spikeSettings).generate()
        else:
            raise ValueError(f"Invalid spikeType")
        similarity = calculate_cosine_similarity(base_template, new_template)
        
        # 指定範囲の中央値からの距離を計算
        target = (templateSimilarityControlSettings["max_cosine_similarity"] + templateSimilarityControlSettings["min_cosine_similarity"]) / 2
        current_distance = abs(similarity - target)
        
        # より中央値に近いテンプレートを保持
        if best_template is None or current_distance < abs(best_similarity - target):
            best_similarity = similarity
            best_template = new_template
    
    return best_template
