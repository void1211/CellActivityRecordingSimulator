from pydantic import BaseModel

class Cell(BaseModel):
    id: int
    x: int
    y: int
    z: int
    group: int = 0  # 細胞のグループID（デフォルトは0）
    spikeTimeList: list[int] = []
    spikeAmpList: list[float] = []
    spikeTemp: list[float] = []

    def __str__(self):
        if len(self.spikeTimeList) <= 10:
            spike_display = str(self.spikeTimeList)
        else:
            spike_display = f"{self.spikeTimeList[:5]}...{self.spikeTimeList[-5:]}"
        return f"Cell(id={self.id}, x={self.x}, y={self.y}, z={self.z}, group={self.group}, spikeTimeList={spike_display})"

    def __repr__(self):
        return self.__str__()
        

        