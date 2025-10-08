from typing import List
import numpy as np
import json
from pathlib import Path
from .Settings import Settings
from .Unit import Unit
from .Contact import Contact
from spikeinterface.core import NumpyRecording
from probeinterface import Probe
from tqdm import tqdm


class CarsObject:
    def __init__(
    self, 
    settings: Settings=None, 
    units: List[Unit]=None, 
    contacts: List[Contact]=None,
    noise_units: List[Unit]=None,
    ):
        self._settings = settings
        self._units = units
        self._contacts = contacts
        self._noise_units = noise_units

    def __str__(self):
        if self._noise_units is None:
            return f"{len(self._units)} units - {len(self._contacts)} ch - no noise units"
        else:
            return f"{len(self._units)} units - {len(self._contacts)} ch - using noise units"
    
    def __repr__(self):
        return self.__str__()

    @property
    def settings(self):
        return self._settings

    @property
    def units(self):
        return self._units

    @property
    def contacts(self):
        return self._contacts

    @property
    def noise_units(self):
        return self._noise_units

    def to_dict(self):
        return {
            "settings": self._settings.to_dict(),
            "units": [unit.to_dict() for unit in self._units],
            "contacts": [contact.to_dict() for contact in self._contacts],
            "noise_units": [noise_unit.to_dict() for noise_unit in self._noise_units],
            }

    @classmethod
    def from_dict(cls, data: dict) -> "CarsObject":
        settings = Settings.from_dict(data["settings"])
        units = [Unit.from_dict(unit) for unit in data["units"]]
        contacts = [Contact.from_dict(contact) for contact in data["contacts"]]
        noise_units = [Unit.from_dict(noise_unit) for noise_unit in data["noise_units"]]
        return cls(
            settings=settings,
            units=units,
            contacts=contacts,
            noise_units=noise_units,
        )

    def save_npz(self, filepath: Path):
        """
        CarsObjectをnpz形式で保存する
        
        Args:
            filepath (Path): 保存先のファイルパス
        """
        data_dict = self.to_dict()       
        np.savez(filepath, **data_dict)


    @classmethod
    def load_npz(cls, file_path: Path) -> "CarsObject":
        """
        npz形式からCarsObjectを読み込む
        
        Args:
            file_path (Path): 読み込み元のファイルパス
            
        Returns:
            CarsObject: 読み込まれたCarsObject
        """
        # npzファイルを読み込み
        with np.load(file_path, allow_pickle=True) as data:
            data_dict = {
                "settings": data["settings"].item(),
                "units": data["units"],
                "noise_units": data["noise_units"],
                "contacts": data["contacts"],
            }
            print(f"CarsObject loaded from {file_path}")
            return cls.from_dict(data_dict)


    def get_NumpyRecording(self, t_starts: List[float]=[0]) -> NumpyRecording:
        """
        t_starts: List[float]
        """
        traces = []
        channel_ids = []
        for contact in self._contacts:
            traces.append(contact.get_signal("raw"))
            channel_ids.append(contact.id)
        traces = np.array(traces).T
        channel_ids = np.array(channel_ids)
        
        fs = self._settings.to_dict()["baseSettings"]["fs"]
            
        recording = NumpyRecording(
            traces,
            fs,
            t_starts=t_starts,
            channel_ids=channel_ids
            )
        return recording

    def get_units_position(self, as_array: bool=False) -> List[List[float]]:
        units_position = []
        for unit in self._units:
            units_position.append([unit.x, unit.y, unit.z])
        if as_array:
            return np.array(units_position)
        return units_position

