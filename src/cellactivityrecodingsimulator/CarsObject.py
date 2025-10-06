from typing import List
import numpy as np
import json
from pathlib import Path
from .Settings import Settings
from .Cell import Cell
from .Site import Site
from spikeinterface.core import NumpyRecording
from tqdm import tqdm


class CarsObject:
    def __init__(
    self, 
    settings: Settings=None, 
    cells: List[Cell]=None, 
    sites: List[Site]=None,
    noise_cells: List[Cell]=None,
    ):
        self._settings = settings
        self._cells = cells
        self._sites = sites
        self._noise_cells = noise_cells

    def __str__(self):
        if self._noise_cells is None:
            return f"{len(self._cells)} units - {len(self._sites)} ch - no noise units"
        else:
            return f"{len(self._cells)} units - {len(self._sites)} ch - using noise units"
    
    def __repr__(self):
        return self.__str__()

    @property
    def settings(self):
        return self._settings

    @property
    def cells(self):
        return self._cells

    @property
    def sites(self):
        return self._sites

    @property
    def noise_cells(self):
        return self._noise_cells

    def to_dict(self):
        return {
            "settings": self._settings.to_dict(),
            "cells": [cell.to_dict() for cell in self._cells],
            "sites": [site.to_dict() for site in self._sites],
            "noise_cells": [noise_cell.to_dict() for noise_cell in self._noise_cells],
            }

    def from_dict(self, data: dict) -> "CarsObject":
        self._settings = Settings.from_dict(data["settings"])
        self._cells = [Cell.from_dict(cell) for cell in data["cells"]]
        self._sites = [Site.from_dict(site) for site in data["sites"]]
        self._noise_cells = [Cell.from_dict(noise_cell) for noise_cell in data["noise_cells"]]
        return self

    def get_NumpyRecording(self, t_starts: List[float]=[0]) -> NumpyRecording:
        """
        t_starts: List[float]
        """
        traces = []
        channel_ids = []
        for site in self._sites:
            traces.append(site.get_signal("raw"))
            channel_ids.append(site.id)
        traces = np.array(traces).T
        recording = NumpyRecording(
            traces,
            self._settings.to_dict()["baseSettings"]["fs"],
            t_starts=t_starts,
            channel_ids=channel_ids
            )
        return recording

    def save_npz(self, filepath: Path):
        """
        CarsObjectをnpz形式で保存する
        
        Args:
            filepath (Path): 保存先のファイルパス
        """
        data_dict = self.to_dict()       
        np.savez(filepath, **data_dict)


    def load_npz(self, file_path: Path):
        """
        npz形式からCarsObjectを読み込む
        
        Args:
            file_path (Path): 読み込み元のファイルパス
            
        Returns:
            CarsObject: 読み込まれたCarsObject
        """
        # npzファイルを読み込み
        data = np.load(file_path, allow_pickle=True)
        
        # 設定を復元
        settings = Settings(data['settings'])
        
        # セルデータを復元
        cells = []
        if 'cells' in data:
            for cell in tqdm(data['cells'], desc="Loading cells", total=len(data['cells'])):
                cells.append(Cell.from_dict(cell))
        else:
            cells = []
        
        # ノイズセルデータを復元
        if 'noise_cells' in data:
            for noise_cell in tqdm(data['noise_cells'], desc="Loading noise cells", total=len(data['noise_cells'])):
                noise_cells.append(Cell.from_dict(noise_cell))
        else:
            noise_cells = []
        
        # サイトデータを復元
        sites = []
        if 'sites' in data:
            for site in tqdm(data['sites'], desc="Loading sites", total=len(data['sites'])):
                sites.append(Site.from_dict(site))
        else:
            sites = []

        self._settings = settings
        self._cells = cells
        self._sites = sites
        self._noise_cells = noise_cells
        
        print(f"CarsObject loaded from {file_path}")
        return self
