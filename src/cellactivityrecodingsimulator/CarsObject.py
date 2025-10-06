from typing import List
import numpy as np
import json
from pathlib import Path
from .Settings import Settings
from .Cell import Cell
from .Site import Site
from spikeinterface.core import NumpyRecording


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
            "cells": {
                "id": [cell.id for cell in self._cells],
                "position": [[cell.x, cell.y, cell.z] for cell in self._cells],
                "spikeTime": [cell.spikeTimeList for cell in self._cells],
                "amplitude": [cell.spikeAmpList for cell in self._cells],
                "template": [cell.spikeTemp for cell in self._cells],
                "group": [cell.group for cell in self._cells]
            },
            "sites": {
                "id": [site.id for site in self._sites],
                "position": [[site.x, site.y, site.z] for site in self._sites],
                "recording": {
                    "raw": [site.get_signal("raw") for site in self._sites],
                    "noise": [site.get_signal("noise") for site in self._sites],
                    "filtered": [site.get_signal("filtered", fs=self._settings.to_dict()["baseSettings"]["fs"]) for site in self._sites],
                    "power": [site.get_signal("power") for site in self._sites],
                    "drift": [site.get_signal("drift") for site in self._sites],
                    "background": [site.get_signal("background") for site in self._sites],
                    "spike": [site.get_signal("spike") for site in self._sites],
                },
            },
            "noise_cells": {
                "id": [noise_cell.id for noise_cell in self._noise_cells],
                "position": [[noise_cell.x, noise_cell.y, noise_cell.z] for noise_cell in self._noise_cells],
                "spikeTime": [noise_cell.spikeTimeList for noise_cell in self._noise_cells],
                "amplitude": [noise_cell.spikeAmpList for noise_cell in self._noise_cells],
                "template": [noise_cell.spikeTemp for noise_cell in self._noise_cells],
                "group": [noise_cell.group for noise_cell in self._noise_cells]
            }
        }

    def from_dict(data: dict) -> "CarsObject":
        return CarsObject(
            settings=Settings.from_dict(data["settings"]),
            cells=[Cell.from_dict(cell) for cell in data["cells"]],
            sites=[Site.from_dict(site) for site in data["sites"]],
            noise_cells=[Cell.from_dict(noise_cell) for noise_cell in data["noise_cells"]]
        )

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

    def save_npz(self, filepath: str):
        """
        CarsObjectをnpz形式で保存する
        
        Args:
            filepath (str): 保存先のファイルパス
        """
        data_dict = self.to_dict()
        
        # 保存用の辞書を作成
        save_dict = {}
        
        # 設定をJSON文字列として保存
        save_dict['settings'] = json.dumps(data_dict['settings'])
        
        # セルデータを保存
        if data_dict['cells']['id']:
            save_dict['cell_ids'] = np.array(data_dict['cells']['id'])
            save_dict['cell_positions'] = np.array(data_dict['cells']['position'])
            save_dict['cell_groups'] = np.array(data_dict['cells']['group'])
            
            # スパイクデータを保存（可変長配列のためリストとして保存）
            spike_times_list = []
            spike_amplitudes_list = []
            spike_templates_list = []
            
            for i, (times, amps, temps) in enumerate(zip(
                data_dict['cells']['spikeTime'],
                data_dict['cells']['amplitude'],
                data_dict['cells']['template']
            )):
                if len(times) > 0:
                    spike_times_list.extend(times)
                    spike_amplitudes_list.extend(amps)
                    spike_templates_list.extend([i] * len(times))
            
            if spike_times_list:
                save_dict['spike_times'] = np.array(spike_times_list)
                save_dict['spike_amplitudes'] = np.array(spike_amplitudes_list)
                save_dict['spike_templates'] = np.array(spike_templates_list)
        
        # ノイズセルデータを保存
        if data_dict['noise_cells']['id']:
            save_dict['noise_cell_ids'] = np.array(data_dict['noise_cells']['id'])
            save_dict['noise_cell_positions'] = np.array(data_dict['noise_cells']['position'])
            
            # ノイズセルのスパイクデータ
            noise_spike_times_list = []
            noise_spike_amplitudes_list = []
            noise_spike_templates_list = []
            
            for i, (times, amps, temps) in enumerate(zip(
                data_dict['noise_cells']['spikeTime'],
                data_dict['noise_cells']['amplitude'],
                data_dict['noise_cells']['template']
            )):
                if len(times) > 0:
                    noise_spike_times_list.extend(times)
                    noise_spike_amplitudes_list.extend(amps)
                    noise_spike_templates_list.extend([i] * len(times))
            
            if noise_spike_times_list:
                save_dict['noise_cell_spike_times'] = np.array(noise_spike_times_list)
                save_dict['noise_cell_spike_amplitudes'] = np.array(noise_spike_amplitudes_list)
                save_dict['noise_cell_spike_templates'] = np.array(noise_spike_templates_list)
        
        # サイトデータを保存
        if data_dict['sites']['id']:
            save_dict['site_ids'] = np.array(data_dict['sites']['id'])
            save_dict['site_positions'] = np.array(data_dict['sites']['position'])
            
            # 信号データを保存
            recording = data_dict['sites']['recording']
            for signal_type in ['raw', 'noise', 'filtered', 'power', 'drift', 'background', 'spike']:
                if signal_type in recording and recording[signal_type]:
                    # 各サイトの信号を結合
                    signals = np.array(recording[signal_type])
                    save_dict[f'signal{signal_type.capitalize()}'] = signals
        
        # npzファイルとして保存
        np.savez_compressed(filepath, **save_dict)
        print(f"CarsObject saved to {filepath}")
def load_npz(cls, filepath: str):
    """
    npz形式からCarsObjectを読み込む
    
    Args:
        filepath (str): 読み込み元のファイルパス
        
    Returns:
        CarsObject: 読み込まれたCarsObject
    """
    # npzファイルを読み込み
    data = np.load(filepath, allow_pickle=True)
    
    # 設定を復元
    settings_dict = json.loads(data['settings'].item())
    settings = Settings.from_dict(settings_dict)
    
    # セルデータを復元
    cells = []
    if 'cell_ids' in data:
        cell_ids = data['cell_ids']
        cell_positions = data['cell_positions']
        cell_groups = data['cell_groups']
        
        # スパイクデータを復元
        spike_times = data.get('spike_times', np.array([]))
        spike_amplitudes = data.get('spike_amplitudes', np.array([]))
        spike_templates = data.get('spike_templates', np.array([]))
        
        for i, (cell_id, pos, group) in enumerate(zip(cell_ids, cell_positions, cell_groups)):
            # このセルに属するスパイクを抽出
            cell_spike_mask = spike_templates == i
            cell_spike_times = spike_times[cell_spike_mask]
            cell_spike_amplitudes = spike_amplitudes[cell_spike_mask]
            
            # テンプレートは仮で作成（実際の実装に応じて調整）
            template = np.zeros((10, 10))  # 仮のテンプレート
            
            cell = Cell(
                id=cell_id,
                x=pos[0], y=pos[1], z=pos[2],
                spikeTimeList=cell_spike_times.tolist(),
                spikeAmpList=cell_spike_amplitudes.tolist(),
                spikeTemp=template,
                group=group
            )
            cells.append(cell)
    
    # ノイズセルデータを復元
    noise_cells = []
    if 'noise_cell_ids' in data:
        noise_cell_ids = data['noise_cell_ids']
        noise_cell_positions = data['noise_cell_positions']
        
        noise_spike_times = data.get('noise_cell_spike_times', np.array([]))
        noise_spike_amplitudes = data.get('noise_cell_spike_amplitudes', np.array([]))
        noise_spike_templates = data.get('noise_cell_spike_templates', np.array([]))
        
        for i, (cell_id, pos) in enumerate(zip(noise_cell_ids, noise_cell_positions)):
            cell_spike_mask = noise_spike_templates == i
            cell_spike_times = noise_spike_times[cell_spike_mask]
            cell_spike_amplitudes = noise_spike_amplitudes[cell_spike_mask]
            
            template = np.zeros((10, 10))  # 仮のテンプレート
            
            noise_cell = Cell(
                id=cell_id,
                x=pos[0], y=pos[1], z=pos[2],
                spikeTimeList=cell_spike_times.tolist(),
                spikeAmpList=cell_spike_amplitudes.tolist(),
                spikeTemp=template,
                group=0  # ノイズセルはグループ0
            )
            noise_cells.append(noise_cell)
    
    # サイトデータを復元
    sites = []
    if 'site_ids' in data:
        site_ids = data['site_ids']
        site_positions = data['site_positions']
        
        for i, (site_id, pos) in enumerate(zip(site_ids, site_positions)):
            # 各信号タイプのデータを復元
            signals = {}
            for signal_type in ['raw', 'noise', 'filtered', 'power', 'drift', 'background', 'spike']:
                signal_key = f'signal{signal_type.capitalize()}'
                if signal_key in data:
                    signals[signal_type] = data[signal_key][i]
            
            site = Site(
                id=site_id,
                x=pos[0], y=pos[1], z=pos[2],
                signals=signals
            )
            sites.append(site)
    
    # CarsObjectを作成
    cars_obj = cls(
        settings=settings,
        cells=cells,
        sites=sites,
        noise_cells=noise_cells if noise_cells else None
    )
    
    print(f"CarsObject loaded from {filepath}")
    return cars_obj
