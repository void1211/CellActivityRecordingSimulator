from typing import List

class BaseObject():
    def __init__(self):
        """
        x: float
        y: float
        z: float
        """
        self.x = 0
        self.y = 0
        self.z = 0
        self.set_position(self.x, self.y, self.z)

    def __str__(self):
        return f"BaseObject[{self.x}, {self.y}, {self.z}]"

    def __repr__(self):
        return self.__str__()

    def set_position(self, x: float, y: float, z: float = 0):
        """
        x: float
        y: float
        z: float
        """
        self.x = x
        self.y = y
        self.z = z

    def get_position(self) -> list[float]:
        return [self.x, self.y, self.z]
