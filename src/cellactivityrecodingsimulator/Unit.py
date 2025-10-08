from .BaseObject import BaseObject
import numpy as np
from typing import Optional, List
from .Template import BaseTemplate

class Unit(BaseObject):
    def __init__(self, id: int = 0, group: int = 0, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None):
        """
        Unitオブジェクトを初期化します。

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
        return f"Unit(id={self.id}, x={self.x}, y={self.y}, z={self.z}, group={self.group})"

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
    @classmethod
    def generate(cls, settings: dict, id: int, group: int, x: float, y: float, z: float) -> 'Unit':
        """設定に基づいてユニットを生成する"""
        unit = cls(id=id, group=group, x=x, y=y, z=z)
        unit.spikeTimeList = cls._make_spike_time(settings)
        unit.spikeAmpList = [cls._choice_spike_amplitude(settings) for _ in unit.spikeTimeList]
        
        if settings["spikeSettings"]["spikeType"] in ["gabor", "exponential"]:
            unit.spikeTemp = BaseTemplate.generate(settings)
        
        return unit

    @classmethod
    def _make_spike_time(cls, settings: dict) -> List[int]:
        """スパイク時間を生成する"""
        duration = settings["baseSettings"]["duration"]
        fs = settings["baseSettings"]["fs"]
        rp = settings["spikeSettings"]["refractoryPeriod"]
        rate = settings["spikeSettings"]["rate"]
        duration_samples = int(duration * fs)

        if rp > 0:
            # 不応期ありの場合
            spike_times, current_time = [], 0.0
            
            while current_time < duration_samples:
                current_time += rp * fs / 1000 + np.random.exponential(1 / rate) * fs
                if current_time < duration_samples:
                    spike_times.append(int(current_time))
            
            return spike_times
        else:
            # 不応期なしの場合
            isi = np.ceil(np.random.exponential(1 / rate, size=10000) * fs)
            spike_times = np.cumsum(isi)
            return spike_times[spike_times < duration_samples].tolist()

    @classmethod
    def _choice_spike_amplitude(cls, settings: dict) -> float:
        """スパイク振幅をランダムに選択する"""
        amp_range = settings["spikeSettings"]
        return np.random.uniform(amp_range["amplitudeMin"], amp_range["amplitudeMax"])

    def to_dict(self):
        return {
            "id": self.id,
            "group": self.group,
            "position": [self.x, self.y, self.z],
            "spikeTime": self.spikeTimeList,
            "amplitude": self.spikeAmpList,
            "template": self.spikeTemp.template,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        """辞書からUnitオブジェクトを作成する"""
        from .Template import GaborTemplate
        
        # 座標の取得（positionまたは個別のx,y,z）
        pos = data.get("position", [0, 0, 0])
        unit = cls(
            id=data.get("id", 0),
            group=data.get("group", 0),
            x=pos[0] if "position" in data else data.get("x", 0),
            y=pos[1] if "position" in data else data.get("y", 0),
            z=pos[2] if "position" in data else data.get("z", 0),
        )
        
        # スパイク情報を設定
        unit.spikeTimeList = data.get("spikeTime", [])
        unit.spikeAmpList = data.get("amplitude", [])
        
        # テンプレートを設定
        template_data = data.get("template", [])
        template_array = np.array(template_data) if len(template_data) > 0 else np.array([0.0])
        unit.spikeTemp = GaborTemplate(template=template_array)
        
        return unit
