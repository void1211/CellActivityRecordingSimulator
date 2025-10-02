import numpy as np

class BaseSignal:
    def __init__(self, fs: float, duration: float):
        
        self._fs = fs
        self._duration = duration
        self._data = np.zeros(int(duration * fs))

    def __repr__(self):
        return f"BaseSignal(fs={self.fs}, duration={self.duration})"

    def __str__(self):
        return self.__repr__()

    @property
    def fs(self):
        return self._fs
    
    @property
    def duration(self):
        return self._duration
    
    @property
    def data(self):
        return self._data
    
    @fs.setter
    def fs(self, fs: float):
        self._check_fs(fs)
        self._fs = fs

    @duration.setter
    def duration(self, duration: float):
        self._check_duration(duration)
        self._duration = duration

    @data.setter
    def data(self, data: np.ndarray):
        self._check_data(data)
        self._data = data

    def _check_fs(self, fs: float):
        assert fs > 0, "fs must be greater than 0"

    def _check_duration(self, duration: float):
        assert duration > 0, "duration must be greater than 0"

    def _check_data(self, data: np.ndarray):
        assert data.shape[0] == self.duration * self.fs, "data must be equal to duration * fs"
        
        