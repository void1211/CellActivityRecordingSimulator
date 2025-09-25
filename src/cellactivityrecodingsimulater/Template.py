import numpy as np
from spikeinterface.core.generate import generate_single_fake_waveform
import logging

from .calculate import calculate_cosine_similarity

class BaseTemplate:
    def __init__(self, **template_parameters):
        self.__template = None
        self.__template_parameters = template_parameters
    
    @property
    def template_parameters(self) -> dict:
        return self.__template_parameters
    
    @template_parameters.setter
    def template_parameters(self, template_parameters: dict):
        self.__template_parameters = template_parameters

    @property
    def template(self) -> np.ndarray:
        return self.__template
    
    @template.setter
    def template(self, template: np.ndarray):
        self.__template = template

    def _check_parameters(self, **template_parameters):
        pass

    def _choose_value(self, parameter_name:str):
        param_value = self.__template_parameters.get(parameter_name)
        if param_value is None:
            raise ValueError(f"Parameter '{parameter_name}' is not provided")
        if len(param_value) == 1:
            return param_value[0]
        else:
            return self._choose_value_from_list(parameter_name)

    def _choose_value_from_list(self, parameter_name:str):
        if self.__template_parameters.get("randType") == "list":
            return np.random.choice(self.__template_parameters.get(parameter_name))
        elif self.__template_parameters.get("randType") == "range":
            return np.random.uniform(self.__template_parameters.get(parameter_name)[0], self.__template_parameters.get(parameter_name)[1])
        else:
            raise ValueError(f"Invalid randType")

class GaborTemplate(BaseTemplate):
    def __init__(self, fs: float, **template_parameters):
        super().__init__(**template_parameters)

        self.__fs = fs
        self.__width = template_parameters.get("spikeWidth")
        self.__sigma = self._choose_value("gaborSigma")
        self.__f0 = self._choose_value("gaborf0")
        self.__theta = self._choose_value("gabortheta")

    def generate(self) -> np.ndarray:
        self._check_parameters(self.__sigma, self.__f0, self.__theta)
        
        self.template = self._gabor()
        return self.template

    def _check_parameters(self, sigma: float, f0: float, theta: float):
        assert isinstance(sigma, float), "sigma must be a float"
        assert isinstance(f0, float), "f0 must be a float"
        assert isinstance(theta, float), "theta must be a float"
    
    def _gabor(self) -> np.ndarray:
        """ガボール関数を生成する"""
        x = np.linspace(-self.__width / 2, self.__width / 2, int(self.__width * self.__fs / 1000))
        x = x / 1000
        sigma_sec = self.__sigma / 1000
        gabortheta_rad = self.__theta * np.pi / 180
        
        y = np.exp(-x**2 / (2 * sigma_sec**2)) * np.cos(2 * np.pi * self.__f0 * x + gabortheta_rad)
        y = y / np.max(np.abs(y))
        return y

class ExponentialTemplate(BaseTemplate):
    def __init__(self, fs: float, **template_parameters):
        super().__init__(**template_parameters)
        self.__fs = fs
        self.__ms_before = self._choose_value("ms_before")
        self.__ms_after = self._choose_value("ms_after")
        self.__negative_amplitude = self._choose_value("negative_amplitude")
        self.__positive_amplitude = self._choose_value("positive_amplitude")
        self.__depolarization_ms = self._choose_value("depolarization_ms")
        self.__repolarization_ms = self._choose_value("repolarization_ms")
        self.__recovery_ms = self._choose_value("recovery_ms")
        self.__smooth_ms = self._choose_value("smooth_ms")
        
    def _check_parameters(self, ms_before: float, ms_after: float, negative_amplitude: float, positive_amplitude: float, depolarization_ms: float, repolarization_ms: float, recovery_ms: float, smooth_ms: float):
        assert isinstance(ms_before, float), "ms_before must be a float"
        assert isinstance(ms_after, float), "ms_after must be a float"
        assert isinstance(negative_amplitude, float), "negative_amplitude must be a float"
        assert isinstance(positive_amplitude, float), "positive_amplitude must be a float"
        assert isinstance(depolarization_ms, float), "depolarization_ms must be a float"
        assert isinstance(repolarization_ms, float), "repolarization_ms must be a float"
        assert isinstance(recovery_ms, float), "recovery_ms must be a float"
        assert isinstance(smooth_ms, float), "smooth_ms must be a float"

    def generate(self) -> np.ndarray:
        self._check_parameters(self.__ms_before, self.__ms_after, self.__negative_amplitude, self.__positive_amplitude, self.__depolarization_ms, self.__repolarization_ms, self.__recovery_ms, self.__smooth_ms)
        template = generate_single_fake_waveform(
        sampling_frequency=self.__fs,
        ms_before=self.__ms_before,
        ms_after=self.__ms_after,
        negative_amplitude=self.__negative_amplitude,
        positive_amplitude=self.__positive_amplitude,
        depolarization_ms=self.__depolarization_ms,
        repolarization_ms=self.__repolarization_ms,
        recovery_ms=self.__recovery_ms,
        smooth_ms=self.__smooth_ms)    
        # ピークを絶対値１に調整
        self.template = template / np.max(np.abs(template))
        return self.template
    

def make_similar_templates(fs: float, num_cells: int, min_cosine_similarity: float, max_cosine_similarity: float, similarity_control_attempts: int, **template_parameters) -> list[list[float]]:
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
            if template_parameters.get("spikeType") == "gabor":
                template = GaborTemplate(fs, **template_parameters).generate()
            elif template_parameters.get("spikeType") == "exponential":
                template = ExponentialTemplate(fs, **template_parameters).generate()
            else:
                raise ValueError(f"Invalid spikeType")
            templates.append(template)
            logging.info(f"基準テンプレートを生成しました")
        else:
            # 2番目以降の細胞は類似度制御されたテンプレートを生成
            template = make_similar_template(fs, templates[0], min_cosine_similarity, max_cosine_similarity, similarity_control_attempts, **template_parameters)
            templates.append(template)
            similarity = calculate_cosine_similarity(templates[0], template)
            logging.info(f"{i}番目のテンプレートを生成しました（類似度: {similarity:.3f})")
    
    return templates

def make_similar_template(fs, base_template: list[float], min_similarity: float, max_similarity: float, max_attempts: int, **template_parameters) -> list[float]:
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
        if template_parameters.get("spikeType") == "gabor":
            new_template = GaborTemplate(fs, **template_parameters).generate()
        elif template_parameters.get("spikeType") == "exponential":
            new_template = ExponentialTemplate(fs, **template_parameters).generate()
        else:
            raise ValueError(f"Invalid spikeType")
        
        # 類似度を計算
        similarity = calculate_cosine_similarity(base_template, new_template)
        
        # 類似度が指定範囲内にあるかチェック
        if min_similarity <= similarity <= max_similarity:
            return new_template
        
        # 反転させたテンプレートもチェック
        inverted_template = -1.0 * new_template
        inverted_similarity = calculate_cosine_similarity(base_template, inverted_template)
        
        # 反転させたテンプレートが指定範囲内にあるかチェック
        if min_similarity <= inverted_similarity <= max_similarity:
            return inverted_template
        
    
    # 最大試行回数に達した場合は、最も近いテンプレートを返す
    logging.warning(f"類似度制御の最大試行回数（{max_attempts}）に達しました。最も範囲に近いテンプレートを使用します。")
    
    best_template = None
    best_similarity = -1.0
    
    for _ in range(10):  # 最後の10回の試行
        if template_parameters.get("spikeType") == "gabor":
            new_template = GaborTemplate(fs, **template_parameters).generate()
        elif template_parameters.get("spikeType") == "exponential":
            new_template = ExponentialTemplate(fs, **template_parameters).generate()
        else:
            raise ValueError(f"Invalid spikeType")
        similarity = calculate_cosine_similarity(base_template, new_template)
        
        # 指定範囲の中央値からの距離を計算
        target = (max_similarity + min_similarity) / 2
        current_distance = abs(similarity - target)
        
        # より中央値に近いテンプレートを保持
        if best_template is None or current_distance < abs(best_similarity - target):
            best_similarity = similarity
            best_template = new_template
    
    return best_template