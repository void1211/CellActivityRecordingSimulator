

class BaseObject():
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        """
        x: float
        y: float
        z: float
        """
        assert self._check_position(x, y, z), "Invalid position"
        self.set_position(x, y, z)
    
    def __str__(self):
        return f"BaseObject(x={self._x}, y={self._y}, z={self._z})"
    
    def __repr__(self):
        return self.__str__()

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        assert self._check_position(value, self._y, self._z), "Invalid position"
        self.set_position(value, self._y, self._z)
    
    @property
    def y(self):
        return self._y
    
    @y.setter
    def y(self, value):
        assert self._check_position(self._x, value, self._z), "Invalid position"
        self.set_position(self._x, value, self._z)
    
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        assert self._check_position(self._x, self._y, value), "Invalid position"
        self.set_position(self._x, self._y, value)

    def _check_position(self, x: float, y: float, z: float):
        if isinstance(x, int):
            x = float(x)
        if isinstance(y, int):
            y = float(y)
        if isinstance(z, int):
            z = float(z)
        
        if not isinstance(x, float):
            return False
        if not isinstance(y, float):
            return False
        if not isinstance(z, float):
            return False
        return True

    def set_position(self, x: float, y: float, z: float = 0):
        """
        x: float
        y: float
        z: float
        """
        assert self._check_position(x, y, z), "Invalid position"
        self._x = x
        self._y = y
        self._z = z

    def from_dict(self, data):
        self._x = data["x"]
        self._y = data["y"]
        self._z = data["z"]
        return self
