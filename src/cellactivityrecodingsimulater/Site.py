from pydantic import BaseModel

class Site(BaseModel):
    id: int
    x: int
    y: int
    z: int
    signalRaw: list[float] = []
    signalFiltered: list[float] = []
    signalNoise: list[float] = []
    signalDrift: list[float] = []
    signalPowerNoise: list[float] = []
    signalBGNoise: list[float] = []
    
    def __str__(self):
        return f"Site(id={self.id}, x={self.x}, y={self.y}, z={self.z})"

    def __repr__(self):
        return self.__str__()
    