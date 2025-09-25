from .BaseObject import BaseObject

class Cell(BaseObject):
    def __init__(self, id: int, group: int = 0, **kwargs):
        super().__init__(**kwargs)
        assert self._check_id(id), "Invalid id"
        assert self._check_group(group), "Invalid group"
        self._id = id
        self._group = group
        self._spikeTimeList = []
        self._spikeAmpList = []
        self._spikeTemp = []

    def __str__(self):
        if len(self.spikeTimeList) <= 10:
            spike_display = str(self.spikeTimeList)
        else:
            spike_display = f"{self.spikeTimeList[:5]}...{self.spikeTimeList[-5:]}"
        return f"Cell(id={self.id}, x={self.x}, y={self.y}, z={self.z}, group={self.group}, spikeTimeList={spike_display})"

    def __repr__(self):
        return self.__str__()

    @property
    def group(self):
        return self._group
    
    @group.setter
    def group(self, value):
        assert self._check_group(value), "Invalid group"
        self._group = value

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        assert self._check_id(value), "Invalid id"
        self._id = value

    def _check_id(self, value):
        if not isinstance(value, int):
            return False
        return True

    def _check_group(self, value):
        if not isinstance(value, int):
            return False
        return True

    def from_dict(self, json_data):
        super().from_dict(json_data)
        self._id = json_data["id"]
        self._group = json_data["group"]
        return self

        