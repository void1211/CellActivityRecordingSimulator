from .BaseSignal import BaseSignal
import numpy as np
from .Cell import Cell
from .Site import Site
from .generate import make_noise_cells

class RandomNoise(BaseSignal):
    def __init__(self, fs: float, duration: float, noiseAmp: float=1.0, loc: float=0.0, scale: float=1.0):
        super().__init__(fs, duration)
        self._noiseAmp = noiseAmp
        self._loc = loc
        self._scale = scale

    def __repr__(self):
        return f"RandomNoise(fs={self.fs}, duration={self.duration}, noiseAmp={self.noiseAmp})"

    def __str__(self):
        return self.__repr__()

    @property
    def noiseAmp(self):
        return self._noiseAmp
    
    @noiseAmp.setter
    def noiseAmp(self, noiseAmp: float):
        self._check_noiseAmp(noiseAmp)
        self._noiseAmp = noiseAmp
    
    @property
    def loc(self):
        return self._loc
    
    @loc.setter
    def loc(self, loc: float):
        self._check_loc(loc)
        self._loc = loc
    
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, scale: float):
        self._check_scale(scale)
        self._scale = scale
    
    def _check_noiseAmp(self, noiseAmp: float):
        assert noiseAmp > 0, "noiseAmp must be greater than 0"
    
    def _check_loc(self, loc: float):
        assert loc >= 0, "loc must be greater than or equal to 0"
    
    def _check_scale(self, scale: float):
        assert scale > 0, "scale must be greater than 0"

    def generate(self, noiseType: str):
        if noiseType == "normal":
            self.data = self._normal_noise()
        elif noiseType == "gaussian":
            self.data = self._gaussian_noise()

    def _normal_noise(self):
        return np.random.default_rng().integers(-self.noiseAmp, self.noiseAmp, size=int(self.duration * self.fs)).astype(np.float64)

    def _gaussian_noise(self):
        return np.random.normal(self.loc, self.scale, size=int(self.duration * self.fs)).astype(np.float64) * self.noiseAmp


class ModelNoise(BaseSignal):
    def __init__(self, fs: float, duration: float, **template_parameters):
        super().__init__(fs, duration)
        self._cells = []
        self._template_parameters = template_parameters

    def __repr__(self):
        return f"ModelNoise(fs={self.fs}, duration={self.duration})"

    def __str__(self):
        return self.__repr__()

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cells: list[Cell]):
        self._cells = cells

    def genearate(self, sites: list[Site], margin: float, density: float, inviolableArea: float):
        self._cells = make_noise_cells(self.duration, self.fs, sites, margin, density, inviolableArea, **self._template_parameters)

class DriftNoise(BaseSignal):
    def __init__(self, fs: float, duration: float, driftSettings: dict):
        super().__init__(fs, duration)
        self._driftType = driftSettings["driftType"]
        self._driftAmplitude = driftSettings.get(driftSettings["driftType"])["amplitude"]
        self._driftFrequency = driftSettings.get(driftSettings["driftType"])["frequency"]

    def __repr__(self):
        return f"DriftNoise(fs={self.fs}, duration={self.duration})"

    def __str__(self):
        return self.__repr__()

    @property
    def driftType(self):
        return self._driftType

    @driftType.setter
    def driftType(self, driftType: str):
        self._driftType = driftType

    @property
    def driftAmplitude(self):
        return self._driftAmplitude

    @driftAmplitude.setter
    def driftAmplitude(self, driftAmplitude: float):
        self._driftAmplitude = driftAmplitude

    @property
    def driftFrequency(self):
        return self._driftFrequency

    @driftFrequency.setter
    def driftFrequency(self, driftFrequency: float):
        self._driftFrequency = driftFrequency

    def generate(self):
        """ドリフト信号をシミュレートする"""
        
        signal_length = int(self.duration * self.fs)
        
        if self.driftType == "linear":
            # 線形ドリフト（時間とともに直線的に変化）
            t = np.linspace(0, self.duration, signal_length)
            drift = np.linspace(-self._driftAmplitude/2, self._driftAmplitude/2, signal_length)
            
        elif self.driftType == "exponential":
            # 指数関数的ドリフト（時間とともに指数関数的に変化）
            t = np.linspace(0, self.duration, signal_length)
            drift = self._driftAmplitude * (np.exp(-t / (self.duration / 3)) - 0.5)
            
        elif self.driftType == "oscillatory":
            # 振動的ドリフト（正弦波的な変化）
            t = np.linspace(0, self.duration, signal_length)
            drift = self.driftAmplitude * np.sin(2 * np.pi * self.driftFrequency * t)
            
        elif self.driftType == "random_walk":
            # ランダムウォークドリフト（ランダムな歩行）
            steps = np.random.normal(0, self.driftAmplitude / 100, signal_length)
            drift = np.cumsum(steps)
            # 振幅を制限
            drift = drift - np.mean(drift)
            drift = drift * (self.driftAmplitude / np.max(np.abs(drift)))
            
        elif self.driftType == "step":
            # ステップ状ドリフト（段階的な変化）
            drift = np.zeros(signal_length)
            step_times = np.random.choice(signal_length, size=3, replace=False)
            step_times.sort()
            
            current_level = 0
            for i, step_time in enumerate(step_times):
                if i < len(step_times) - 1:
                    next_step = step_times[i + 1]
                    drift[step_time:next_step] = current_level
                    current_level += np.random.uniform(-self.driftAmplitude/2, self.driftAmplitude/2)
                else:
                    drift[step_time:] = current_level
                    
        else:
            raise ValueError(f"Unknown drift type: {self.driftType}")
        
        return drift.astype(np.float64)

class PowerLineNoise(BaseSignal):
    def __init__(self, fs: float, duration: float, powerLineFrequency: float = 50.0, powerLineAmplitude: float = 20.0):
        super().__init__(fs, duration)
        self._powerLineFrequency = powerLineFrequency
        self._powerLineAmplitude = powerLineAmplitude

    def __repr__(self):
        return f"PowerLineNoise(fs={self.fs}, duration={self.duration})"

    def __str__(self):
        return self.__repr__()

    @property
    def powerLineFrequency(self):
        return self._powerLineFrequency

    @powerLineFrequency.setter
    def powerLineFrequency(self, powerLineFrequency: float):
        self._powerLineFrequency = powerLineFrequency

    @property
    def powerLineAmplitude(self):
        return self._powerLineAmplitude

    @powerLineAmplitude.setter
    def powerLineAmplitude(self, powerLineAmplitude: float):
        self._powerLineAmplitude = powerLineAmplitude

    def generate(self):
        """
        電源ノイズ(50Hz/60Hz)をシミュレートする
        
        Args:
            settings: シミュレーション設定
            powerLineFreq: 電源周波数(Hz, デフォルト50Hz)
        
        Returns:
            np.ndarray: 電源ノイズ信号
        """
        signal_length = int(self.duration * self.fs)
        
        # 時間軸
        t = np.linspace(0, self.duration, signal_length)
        
        # 基本周波数の電源ノイズ
        power_noise = self.powerLineAmplitude * np.sin(2 * np.pi * self.powerLineFrequency * t)
        
        # 高調波を追加（3次、5次、7次）
        harmonics = [3, 5, 7]
        harmonic_amplitudes = [0.3, 0.2, 0.1]  # 基本波に対する比率
        
        for harmonic, amplitude_ratio in zip(harmonics, harmonic_amplitudes):
            harmonic_noise = self.powerLineAmplitude * amplitude_ratio * np.sin(2 * np.pi * self.powerLineFrequency * harmonic * t)
            power_noise += harmonic_noise
        
        # 位相のゆらぎを追加（より現実的な電源ノイズ）
        phase_noise = np.random.normal(0, 0.1, signal_length)  # 小さな位相ノイズ
        power_noise_with_phase = self.powerLineAmplitude * np.sin(2 * np.pi * self.powerLineFrequency * t + phase_noise)
        
        # 基本ノイズと位相ノイズを組み合わせ
        final_power_noise = 0.7 * power_noise + 0.3 * power_noise_with_phase
        
        return final_power_noise.astype(np.float64)
