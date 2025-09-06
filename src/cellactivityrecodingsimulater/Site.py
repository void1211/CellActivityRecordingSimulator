from .BaseObject import BaseObject
import numpy as np

class Site(BaseObject):
    def __init__(self, id: int, **kwargs):
        super().__init__(**kwargs)
        assert self._check_id(id), "Invalid id"
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
    
    def set_signal(self, signal: np.ndarray, signal_type: str):
        assert signal_type in ["spike", "drift", "power", "background"], "Invalid signal type"
        if not isinstance(signal, np.ndarray) and isinstance(signal, list):
            signal = np.array(signal)
        assert isinstance(signal, np.ndarray), "Signal must be a numpy array"

        if signal_type == "spike":
            self._signal_spike = signal
        elif signal_type == "drift":
            self._signal_drift = signal
        elif signal_type == "power":
            self._signal_power = signal
        elif signal_type == "background":
            self._signal_background = signal

    def get_signal(self, signal_type: str):
        assert signal_type in ["spike", "drift", "power", "background", "raw", "filtered", "noise"], "Invalid signal type"
        if signal_type == "spike":
            return self._signal_spike
        elif signal_type == "drift":
            return self._signal_drift
        elif signal_type == "power":
            return self._signal_power
        elif signal_type == "background":
            return self._signal_background
        elif signal_type == "raw":
            return self._make_signal([self.get_signal("spike"), self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")])  
        # elif signal_type == "filtered":
        #     return self._make_signal([self.get_signal("spike"), self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")])
        elif signal_type == "noise":
            return self._make_signal([self.get_signal("drift"), self.get_signal("power"), self.get_signal("background")])

    def _make_signal(self, signals: list[np.ndarray]):
        return sum(np.array(signals))