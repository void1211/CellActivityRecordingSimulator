from .BaseSignal import BaseSignal
import numpy as np

class RandomNoise(BaseSignal):
    def __init__(self):
        self.signal = []

    def __repr__(self):
        return f"RandomNoise: length={self.signal.shape[0]}"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def generate(cls, noiseType: str, settings):
        """ランダムノイズを生成する"""
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        noise = cls()
        if noiseType == "normal":
            noise.signal = cls._normal_noise(settings)
        elif noiseType == "gaussian":
            noise.signal = cls._gaussian_noise(settings)
        return noise

    def _normal_noise(cls, settings: dict):
        fs = settings["baseSettings"]["fs"]
        duration = settings["baseSettings"]["duration"]
        noiseAmp = settings["noiseSettings"]["normal"]["amplitude"]
        signal = np.random.default_rng().integers(-noiseAmp, noiseAmp, size=int(duration * fs)).astype(np.float64)
        return signal

    def _gaussian_noise(cls, settings: dict):
        fs = settings["baseSettings"]["fs"]
        duration = settings["baseSettings"]["duration"]
        noiseAmp = settings["noiseSettings"]["gaussian"]["amplitude"]
        loc = settings["noiseSettings"]["gaussian"]["location"]
        scale = settings["noiseSettings"]["gaussian"]["scale"]
        signal = np.random.normal(loc, scale, size=int(duration * fs)).astype(np.float64) * noiseAmp
        return signal

# class ModelNoise(BaseSignal):
#     def __init__(self, settings: dict, units: list[Unit]):
#         super().__init__(settings["baseSettings"]["fs"], settings["baseSettings"]["duration"])
#         self._units = units
#         # self.signal = generate(settings)

#     def __repr__(self):
#         return f"ModelNoise(fs={self.fs}, duration={self.duration})"

#     def __str__(self):
#         return self.__repr__()

#     @property
#     def units(self):
#         return self._units

#     @units.setter
#     def units(self, units: list[Unit]):
#         self._units = units

#     # def genearate(self, contacts: list[Contact], margin: float, density: float, inviolableArea: float):
#     #     self._units = make_noise_units(self.duration, self.fs, contacts, margin, density, inviolableArea)

class DriftNoise(BaseSignal):
    def __init__(self):

        self.signal = []
    def __repr__(self):
        return f"DriftNoise: length={self.signal.shape[0]}"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def generate(cls, settings):
        """ドリフト信号をシミュレートする"""
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        noise = cls()
        signal_length = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
        
        if settings["driftSettings"]["enable"] == False:
            noise.signal = np.zeros(signal_length)
            return noise

        if settings["driftSettings"]["driftType"] == "linear":
            # 線形ドリフト（時間とともに直線的に変化）
            t = np.linspace(0, settings["baseSettings"]["duration"], signal_length)
            drift = np.linspace(-settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"]/2, settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"]/2, signal_length)
            
        elif settings["driftSettings"]["driftType"] == "exponential":
            # 指数関数的ドリフト（時間とともに指数関数的に変化）
            t = np.linspace(0, settings["baseSettings"]["duration"], signal_length)
            drift = settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"] * (np.exp(-t / (settings["baseSettings"]["duration"] / 3)) - 0.5)
            
        elif settings["driftSettings"]["driftType"] == "oscillatory":
            # 振動的ドリフト（正弦波的な変化）
            t = np.linspace(0, settings["baseSettings"]["duration"], signal_length)
            drift = settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"] * np.sin(2 * np.pi * settings["driftSettings"][settings["driftSettings"]["driftType"]]["frequency"] * t)
            
        elif settings["driftSettings"]["driftType"] == "random_walk":
            # ランダムウォークドリフト（ランダムな歩行）
            steps = np.random.normal(0, settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"] / 100, signal_length)
            drift = np.cumsum(steps)
            # 振幅を制限
            drift = drift - np.mean(drift)
            drift = drift * (settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"] / np.max(np.abs(drift)))
            
        elif settings["driftSettings"]["driftType"] == "step":
            # ステップ状ドリフト（段階的な変化）
            drift = np.zeros(signal_length)
            step_times = np.random.choice(signal_length, size=3, replace=False)
            step_times.sort()
            
            current_level = 0
            for i, step_time in enumerate(step_times):
                if i < len(step_times) - 1:
                    next_step = step_times[i + 1]
                    drift[step_time:next_step] = current_level
                    current_level += np.random.uniform(-settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"]/2, settings["driftSettings"][settings["driftSettings"]["driftType"]]["amplitude"]/2)
                else:
                    drift[step_time:] = current_level
                    
        else:
            raise ValueError(f"Unknown drift type")
        
        noise.signal = drift.astype(np.float64)
        return noise

class PowerLineNoise(BaseSignal):
    def __init__(self):
        self.signal = []

    def __repr__(self):
        return f"PowerLineNoise: length={self.signal.shape[0]}"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def generate(cls, settings):
        """電源ノイズ(50Hz/60Hz)をシミュレートする"""
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        noise = cls()
        signal_length = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
        
        if settings["powerNoiseSettings"]["enable"] == False:
            noise.signal = np.zeros(signal_length)
            return noise

        # 時間軸
        t = np.linspace(0, settings["baseSettings"]["duration"], signal_length)
        
        # 基本周波数の電源ノイズ
        power_noise = settings["powerLineSettings"]["amplitude"] * np.sin(2 * np.pi * settings["powerLineSettings"]["frequency"] * t)
        
        # 高調波を追加（3次、5次、7次）
        harmonics = [3, 5, 7]
        harmonic_amplitudes = [0.3, 0.2, 0.1]  # 基本波に対する比率
        
        for harmonic, amplitude_ratio in zip(harmonics, harmonic_amplitudes):
            harmonic_noise = settings["powerLineSettings"]["amplitude"] * amplitude_ratio * np.sin(2 * np.pi * settings["powerLineSettings"]["frequency"] * harmonic * t)
            power_noise += harmonic_noise
        
        # 位相のゆらぎを追加（より現実的な電源ノイズ）
        phase_noise = np.random.normal(0, 0.1, signal_length)  # 小さな位相ノイズ
        power_noise_with_phase = settings["powerLineSettings"]["amplitude"] * np.sin(2 * np.pi * settings["powerLineSettings"]["frequency"] * t + phase_noise)
        
        # 基本ノイズと位相ノイズを組み合わせ
        final_power_noise = 0.7 * power_noise + 0.3 * power_noise_with_phase
        
        noise.signal = final_power_noise.astype(np.float64)
        return noise
