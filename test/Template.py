import numpy as np
from spikeinterface.core.generate import generate_single_fake_waveform


class BaseTemplate:
    def __init__(self):
        self.__template = None

    @property
    def template(self) -> np.ndarray:
        return self.__template
    
    @template.setter
    def template(self, template: np.ndarray):
        self.__template = template

class GaborTemplate(BaseTemplate):
    def __init__(self, fs: float, width: float, sigma: float, f0: float, theta: float):
        super().__init__()

        self.__fs = fs
        self.__width = width
        self.__sigma = sigma
        self.__f0 = f0
        self.__theta = theta

    def generate(self) -> np.ndarray:
        self._check_parameters(self.__sigma, self.__f0, self.__theta)

        # if self.randType == "list":
        #     sigma = np.random.choice(self._sigma)
        #     f0 = np.random.choice(self._f0)
        #     theta = np.random.choice(self._theta)
        # elif self.randType == "range":
        #     sigma = np.random.uniform(self._sigma[0], self._sigma[1])
        #     f0 = np.random.uniform(self._f0[0], self._f0[1])
        #     theta = np.random.uniform(self.theta[0], self.theta[1])
        
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
    def __init__(self, fs: float, ms_before: float, ms_after: float, negative_amplitude: float, positive_amplitude: float, depolarization_ms: float, repolarization_ms: float, recovery_ms: float, smooth_ms: float, randType: str):
        super().__init__()
        self.__fs = fs
        self.__ms_before = ms_before
        self.__ms_after = ms_after
        self.__negative_amplitude = negative_amplitude
        self.__positive_amplitude = positive_amplitude
        self.__depolarization_ms = depolarization_ms
        self.__repolarization_ms = repolarization_ms
        self.__recovery_ms = recovery_ms
        self.__smooth_ms = smooth_ms
        
    def _check_parameters(self, ms_before: float, ms_after: float, negative_amplitude: float, positive_amplitude: float, depolarization_ms: float, repolarization_ms: float, recovery_ms: float, smooth_ms: float, randType: str):
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
    
    