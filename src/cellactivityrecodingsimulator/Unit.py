from .BaseObject import BaseObject
import numpy as np
from typing import List
from .Template import BaseTemplate

class Unit(BaseObject):
    def __init__(self):
        
        # セッターを使ってバリデーションを適用
        self.id = 0
        self.group = 0

        self.spikeTimeList = []
        self.spikeAmpList = []
        self.templateObject = BaseTemplate()

    def __str__(self):
        return f"Unit{self.id}({self.group}), [{self.x},{self.y},{self.z}]"

    def __repr__(self):
        return self.__str__()

    def set_spike_time(self, spike_time: List[int]=None, settings=None):
        """スパイク時間を設定する"""
        if spike_time is not None:
            self.spikeTimeList = spike_time
        else:
            # Settingsオブジェクトの場合は辞書に変換
            if hasattr(settings, 'to_dict'):
                settings = settings.to_dict()
            self.spikeTimeList = self._make_spike_time(settings)

    def set_amplitudes(self, amplitude: List[float]=None, settings=None):
        """スパイク振幅を設定する"""
        if amplitude is not None:
            self.spikeAmpList = amplitude
        else:
            # Settingsオブジェクトの場合は辞書に変換
            if hasattr(settings, 'to_dict'):
                settings = settings.to_dict()
            self.spikeAmpList = [self._choice_spike_amplitude(settings) for _ in self.spikeTimeList]

    def get_templateObject(self) -> BaseTemplate:
        """テンプレートオブジェクトを取得する"""
        return self.templateObject

    def set_templateObject(self, template: BaseTemplate=None, settings=None):
        """テンプレートオブジェクトを設定する"""
        if template is not None:
            self.templateObject = template
        else:
            self.templateObject = BaseTemplate.generate(settings)

    @classmethod
    def generate(cls, id: int, group: int, x: float, y: float, z: float) -> 'Unit':
        """設定に基づいてユニットを生成する"""
        unit = cls()
        unit.id = id
        unit.group = group
        unit.x = x
        unit.y = y
        unit.z = z
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
            "template": self.templateObject.get_template(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        """辞書からUnitオブジェクトを作成する"""
        from .Template import GaborTemplate
        
        # 座標の取得（positionまたは個別のx,y,z）
        if "position" in data:
            pos = data["position"]
            x, y, z = pos[0], pos[1], pos[2]
        else:
            x = data.get("x", 0)
            y = data.get("y", 0)
            z = data.get("z", 0)
        
        unit = cls.generate(
            id=data.get("id", 0),
            group=data.get("group", 0),
            x=x,
            y=y,
            z=z
        )

        # スパイク情報を設定
        unit.spikeTimeList = data.get("spikeTime", [])
        unit.spikeAmpList = data.get("amplitude", [])
        
        # テンプレートを設定
        if "template" in data and len(data["template"]) > 0:
            unit.templateObject = BaseTemplate().set_template(list(data["template"]))
        else:
            unit.templateObject = BaseTemplate().set_template(np.array([0.0]))
        
        return unit
