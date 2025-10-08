from .Unit import Unit
import numpy as np
from spikeinterface.core import NumpySorting
from pathlib import Path
from typing import Union
import json



class GTUnitsObject:

    def __init__(self, units: list[Unit]):
        self._units = units

    def __str__(self):
        return f"GTUnitss - {len(self._units)} units"

    def __repr__(self):
        return self.__str__()

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = value

    @classmethod
    def generate(
        cls, 
        settings:dict,
        num_column:int=1, 
        num_unit_per_column:int=16, 
        xpitch:float=0, 
        ypitch:float=25.0, 
        y_shift_per_column:list[float]=[0], 
        x_shift:float=0, 
        id_major_order:str="column"
        ):
        units = []
        current_id = 0
        
        if id_major_order == "column":
            # 列優先: 各列を順番に処理
            for column in range(num_column):
                for unit in range(num_unit_per_column):
                    units.append(
                        Unit.generate(
                            settings=settings,
                            id=current_id, 
                            group=0, 
                            x=xpitch * column + x_shift, 
                            y=ypitch * unit + y_shift_per_column[column],
                            z=0
                        ))
                    current_id += 1
        elif id_major_order == "row":
            # 行優先: 各行を順番に処理
            max_units = num_unit_per_column if num_unit_per_column else 0
            for unit in range(max_units):
                for column in range(num_column):
                    if unit < num_unit_per_column:
                        units.append(
                            Unit.generate(
                                settings=settings,
                                id=current_id, 
                                group=0, 
                                x=xpitch * column + x_shift, 
                                y=ypitch * unit + y_shift_per_column[column],
                                z=0
                            ))
                        current_id += 1
        else:
            raise ValueError(f"Invalid id_major_order: {id_major_order}")

        return units

    @classmethod
    def load(cls, object: Union[Path, "GTUnitsObject", dict, None], settings: dict) -> "GTUnitsObject":
        if object is None:
            return cls.default_GTUnitsObject(settings=settings)
        elif isinstance(object, Path):
            return cls.from_json(object)
        elif isinstance(object, GTUnitsObject):
            return object
        elif isinstance(object, dict):
            return cls.from_dict(object)
        else:
            raise ValueError(f"Invalid object: {object}")


    @classmethod
    def from_json(cls, object: Path):
        if isinstance(object, Path):
            if not Path(object).exists():
                raise FileNotFoundError(f"セルファイルが見つかりません: {object}")
        
        if Path(object).stat().st_size == 0:
            raise ValueError(f"セルファイルが空です: {object}")

        units = []

        if isinstance(object, dict):
            junits = object
        else:
            with open(object, "r") as f:
                junits = json.load(f)

        for i in range(len(junits["id"])):
            unit_data = {
                "id": junits["id"][i], 
                "x": junits["x"][i], 
                "y": junits["y"][i], 
                "z": junits["z"][i]
            }
            units.append(Unit.from_dict(unit_data)) 
        return cls(units=units)

    def to_dict(self):
        dict_data = {
            "units": [unit.to_dict() for unit in self._units]
        }
        return dict_data

    @classmethod
    def from_dict(cls, data: dict):
        units = [Unit.from_dict(unit) for unit in data["units"]]
        return cls(units=units)

    def get_units_num(self):
        return len(self._units)

    def save_npz(self, filepath: str):
        data_dict = self.to_dict()
        np.savez_compressed(filepath, **data_dict)

    @classmethod
    def load_npz(cls, filepath: str):
        data_dict = np.load(filepath)
        units = [Unit.from_dict(unit) for unit in data_dict["units"]]
        return cls(units=units)

    def to_Sorting(self, sampling_frequency: float):
        units_dict = {}
        for index, unit in enumerate(self._units):
            units_dict[index] = np.array(unit.spikeTimeList)

        gt_sorting = NumpySorting.from_unit_dict(
            units_dict_list=units_dict,
            sampling_frequency=sampling_frequency,
        )
        return gt_sorting

    @classmethod
    def default_GTUnitsObject(cls, settings: dict):
        units = cls.generate(
            settings=settings,
            num_column=1,
            num_unit_per_column=16,
            xpitch=0,
            ypitch=50,
            y_shift_per_column=[0],
            x_shift=0,
            id_major_order="column",
        )
        return cls(units=units)

