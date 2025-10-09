import numpy as np


class BaseSignal:
    def __init__(self, fs: float, duration: float):
        self._fs = fs
        self._duration = duration
        self._signal = np.zeros(int(duration * fs))

    def __repr__(self):
        return f"BaseSignal(fs={self.fs}, duration={self.duration})"

    def __str__(self):
        return self.__repr__()

