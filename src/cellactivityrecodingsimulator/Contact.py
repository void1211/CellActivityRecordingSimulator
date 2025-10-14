from .BaseObject import BaseObject
import numpy as np
from .tools import filterSignal
from .calculate import calculate_scaled_spike_amplitude, calculate_distance_two_objects
from .Unit import Unit

class Contact(BaseObject):
    def __init__(self):
        super().__init__()
        self.id = 0

        self.signal_spike = []
        self.signal_drift = []
        self.signal_power = []
        self.signal_background = []
    
    def __str__(self):
        return f"Contact{self.id}, [{self.x},{self.y},{self.z}]"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def generate(cls, settings: dict, id: int, x: float, y: float, z: float) -> "Contact":
        contact = cls()
        contact.id = id
        contact.x = x
        contact.y = y
        contact.z = z
        contact.signal_spike = np.zeros(int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"]))
        contact.signal_drift = np.zeros(int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"]))
        contact.signal_power = np.zeros(int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"]))
        contact.signal_background = np.zeros(int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"]))
        return contact

    def set_signal(self, signal_type: str, signal: np.ndarray, dtype: np.dtype=np.int16):
        """
        signal: np.ndarray or list
        signal_type: str

        signal_type: spike, drift, power, background
        """
        # assert isinstance(signal_type, str), "Signal type must be a string"
        assert signal_type in ["spike", "drift", "power", "background", "raw", "noise", "filtered"], "Invalid signal type"

        if isinstance(signal, list):
            signal = np.array(signal, dtype=dtype)
        # assert isinstance(signal, np.ndarray), "Signal must be a numpy array or list"

        if signal_type == "spike":
            self.signal_spike = signal.astype(dtype=dtype)
        elif signal_type == "drift":
            self.signal_drift = signal.astype(dtype=dtype)
        elif signal_type == "power":
            self.signal_power = signal.astype(dtype=dtype)
        elif signal_type == "background":
            self.signal_background = signal.astype(dtype=dtype)
        elif signal_type == "raw":
            self.signal_raw = signal.astype(dtype=dtype)
        elif signal_type == "noise":
            self.signal_noise = signal.astype(dtype=dtype)
        elif signal_type == "filtered":
            self.signal_filtered = signal.astype(dtype=dtype)

    def get_signal(self, signal_type: str, fs: float=None) -> np.ndarray:
        """
        signal_type: spike, drift, power, background, raw, filtered, noise
        return: np.ndarray or list
        """
        assert signal_type in ["spike", "drift", "power", "background", "raw", "filtered", "noise"], "Invalid signal type"
        if signal_type == "spike":
            return np.array(self.signal_spike)
        elif signal_type == "drift":
            return np.array(self.signal_drift)
        elif signal_type == "power":
            return np.array(self.signal_power)
        elif signal_type == "background":
            return np.array(self.signal_background)
        elif signal_type == "raw":
            return np.array(self._make_signal([self.get_signal("spike"), self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")]))  
        elif signal_type == "filtered":
            signal = np.array(self._make_signal([self.get_signal("spike"), self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")]))
            return filterSignal(signal, fs, 300, 3000)
        elif signal_type == "noise":
            return np.array(self._make_signal([self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")]))

    def _make_signal(self, signals: list[np.ndarray]) -> np.ndarray:
        """複数の信号を合成する"""
        # 空でない信号のみをフィルタ
        valid_signals = [s for s in signals if len(s) > 0]
        
        if not valid_signals:
            return np.array([])
        
        # すべての信号が同じ長さであることを確認
        max_length = max(len(s) for s in valid_signals)
        
        # 長さが異なる信号を最大長に揃える（ゼロパディング）
        padded_signals = []
        for s in valid_signals:
            if len(s) < max_length:
                padded = np.zeros(max_length)
                padded[:len(s)] = s
                padded_signals.append(padded)
            else:
                padded_signals.append(s)
        
        return sum(padded_signals)

    def add_spikes(self, unit: Unit, settings):
        """スパイクを信号に追加する"""
        # Settingsオブジェクトの場合は辞書に変換
        if hasattr(settings, 'to_dict'):
            settings = settings.to_dict()
        
        # 信号が空の場合は初期化
        if len(self.signal_spike) == 0:
            duration_samples = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
            signal = np.zeros(duration_samples, dtype=np.float64)
        else:
            signal = self.get_signal("spike").astype(np.float64)

        spikeTemp = unit.templateObject.get_template()
        spikeTimes = unit.spikeTimeList
        spikeAmpList = calculate_scaled_spike_amplitude(
            unit.spikeAmpList, 
            calculate_distance_two_objects(unit, self), 
            settings["spikeSettings"]["attenTime"]
        )
        # ピーク位置を正しく計算（負のピークも考慮）
        if np.min(spikeTemp) < 0 and abs(np.min(spikeTemp)) > abs(np.max(spikeTemp)):
            # 負のピークが主成分の場合
            peak = np.argmin(spikeTemp)
        else:
            # 正のピークが主成分の場合
            peak = np.argmax(spikeTemp)
        
        for spikeTime, spikeAmp in zip(spikeTimes, spikeAmpList):
            start = int(spikeTime - peak)
            end = int(start + len(spikeTemp))
            if not (0 <= start and end <= len(signal)):
                continue
            signal[start:end] += spikeAmp * spikeTemp
        
        self.set_signal("spike", signal)
    
    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "spike": self.get_signal("spike"),
            "drift": self.get_signal("drift"),
            "power": self.get_signal("power"),
            "background": self.get_signal("background"),
        }

    @classmethod    
    def from_dict(cls, data: dict) -> "Contact":
        contact = cls()
        contact.id = data.get("id", None)
        contact.x = data.get("x", None)
        contact.y = data.get("y", None)
        contact.z = data.get("z", None)
        contact.set_signal("spike", data.get("spike", np.array([])))
        contact.set_signal("drift", data.get("drift", np.array([])))
        contact.set_signal("power", data.get("power", np.array([])))
        contact.set_signal("background", data.get("background", np.array([])))
        return contact
