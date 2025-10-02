from .BaseObject import BaseObject
import numpy as np
from .tools import filterSignal

class Site(BaseObject):
    def __init__(self, id: int=0, x: float=None, y: float=None, z: float=None):
        if all(arg is not None for arg in [x, y, z]):
            super().__init__(x=x, y=y, z=z)
        else:
            super().__init__()
        self._id = id

        self._signal_spike = []
        self._signal_drift = []
        self._signal_power = []
        self._signal_background = []

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        assert self._check_id(value), "Invalid id"
        self._id = value
    
    def __str__(self):
        return f"Site(id={self.id}, x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()

    def _check_id(self, value):
        if not isinstance(value, int):
            return False
        return True
    
    def set_signal(self, signal_type: str, signal: np.ndarray):
        """
        signal: np.ndarray or list
        signal_type: str

        signal_type: spike, drift, power, background
        """
        assert isinstance(signal_type, str), "Signal type must be a string"
        assert signal_type in ["spike", "drift", "power", "background"], "Invalid signal type"

        if isinstance(signal, list):
            signal = np.array(signal)
        assert isinstance(signal, np.ndarray), "Signal must be a numpy array or list"

        if signal_type == "spike":
            self._signal_spike = signal
        elif signal_type == "drift":
            self._signal_drift = signal
        elif signal_type == "power":
            self._signal_power = signal
        elif signal_type == "background":
            self._signal_background = signal

    def get_signal(self, signal_type: str, fs: float=None) -> np.ndarray:
        """
        signal_type: spike, drift, power, background, raw, filtered, noise
        return: np.ndarray or list
        """
        assert signal_type in ["spike", "drift", "power", "background", "raw", "filtered", "noise"], "Invalid signal type"
        if signal_type == "spike":
            return np.array(self._signal_spike)
        elif signal_type == "drift":
            return np.array(self._signal_drift)
        elif signal_type == "power":
            return np.array(self._signal_power)
        elif signal_type == "background":
            return np.array(self._signal_background)
        elif signal_type == "raw":
            return np.array(self._make_signal([self.get_signal("spike"), self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")]))  
        elif signal_type == "filtered":
            signal = np.array(self._make_signal([self.get_signal("spike"), self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")]))
            return filterSignal(signal, fs, 300, 3000)
        elif signal_type == "noise":
            return np.array(self._make_signal([self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")]))

    def _make_signal(self, signals: list[np.ndarray]):
        return sum(signals)

    def from_dict(self, data):
        super().from_dict(data)

        self._id = data.get("id", 0)
        return self