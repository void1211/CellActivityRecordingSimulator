from .BaseObject import BaseObject
import numpy as np
from .tools import filterSignal

class Contact(BaseObject):
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

        self._signal_raw = []
        self._signal_noise = []
        self._signal_filtered = []

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        assert self._check_id(value), "Invalid id"
        self._id = value
    
    def __str__(self):
        return f"Contact(id={self.id}, x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()

    def _check_id(self, value):
        if not isinstance(value, int):
            return False
        return True
    
    def set_signal(self, signal_type: str, signal: np.ndarray, dtype: np.dtype=np.int16):
        """
        signal: np.ndarray or list
        signal_type: str

        signal_type: spike, drift, power, background
        """
        assert isinstance(signal_type, str), "Signal type must be a string"
        assert signal_type in ["spike", "drift", "power", "background", "raw", "noise", "filtered"], "Invalid signal type"

        if isinstance(signal, list):
            signal = np.array(signal, dtype=dtype)
        assert isinstance(signal, np.ndarray), "Signal must be a numpy array or list"

        if signal_type == "spike":
            self._signal_spike = signal.astype(dtype=dtype)
        elif signal_type == "drift":
            self._signal_drift = signal.astype(dtype=dtype)
        elif signal_type == "power":
            self._signal_power = signal.astype(dtype=dtype)
        elif signal_type == "background":
            self._signal_background = signal.astype(dtype=dtype)
        elif signal_type == "raw":
            self._signal_raw = signal.astype(dtype=dtype)
        elif signal_type == "noise":
            self._signal_noise = signal.astype(dtype=dtype)
        elif signal_type == "filtered":
            self._signal_filtered = signal.astype(dtype=dtype)

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
    
    def to_dict(self):
        return {
            "id": self.id,
            "position": [self.x, self.y, self.z],
            "spike": self.get_signal("spike"),
            "drift": self.get_signal("drift"),
            "power": self.get_signal("power"),
            "background": self.get_signal("background"),
        }

    @classmethod    
    def from_dict(cls, data: dict) -> "Contact":
        contact = cls(
            id=data.get("id", 0),
            x=data.get("x", 0),
            y=data.get("y", 0),
            z=data.get("z", 0),
        )
        contact.set_signal("spike", data.get("spike", np.array([])))
        contact.set_signal("drift", data.get("drift", np.array([])))
        contact.set_signal("power", data.get("power", np.array([])))
        contact.set_signal("background", data.get("background", np.array([])))
        return contact
