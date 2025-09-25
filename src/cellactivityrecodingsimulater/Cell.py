from .BaseObject import BaseObject
import numpy as np

class Cell(BaseObject):
    def __init__(self, id: int = 0, group: int = 0, x=None, y=None, z=None):
        """
        Cellオブジェクトを初期化します。

        Args:
            id (int): セルのユニークID
            group (int): セルのグループID
            x, y, z: セルの座標
        """
        # 親クラスの__init__を呼び出し
        if all(arg is not None for arg in [x, y, z]):
            super().__init__(x=x, y=y, z=z)
        else:
            super().__init__()
        
        # セッターを使ってバリデーションを適用
        self.id = id
        self.group = group

        self._spikeTimeList = []
        self._spikeAmpList = []
        self._spikeTemp = []

    def __str__(self):
        return f"Cell(id={self.id}, x={self.x}, y={self.y}, z={self.z}, group={self.group})"

    def __repr__(self):
        return self.__str__()

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, value):
        self._check_group(value)
        self._group = value

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._check_id(value)
        self._id = value
    
    @property
    def spikeTimeList(self):
        """スパイク時間リストを取得"""
        return self._spikeTimeList
    
    @spikeTimeList.setter
    def spikeTimeList(self, value):
        """スパイク時間リストを設定"""
        self._spikeTimeList = value
    
    @property
    def spikeAmpList(self):
        """スパイク振幅リストを取得"""
        return self._spikeAmpList
    
    @spikeAmpList.setter
    def spikeAmpList(self, value):
        """スパイク振幅リストを設定"""
        self._spikeAmpList = value
    
    @property
    def spikeTemp(self):
        """スパイクテンプレートを取得"""
        return self._spikeTemp
    
    @spikeTemp.setter
    def spikeTemp(self, value):
        """スパイクテンプレートを設定"""
        self._spikeTemp = value
        
    def _check_id(self, value):
        if not isinstance(value, int):
            raise TypeError("idは整数である必要があります")

    def _check_group(self, value):
        if not isinstance(value, int):
            raise TypeError("groupは整数である必要があります")
    
    def from_dict(self, data: dict):
        super().from_dict(data)
        self.id = data.get("id", 0)
        self.group = data.get("group", 0)
        
        return self