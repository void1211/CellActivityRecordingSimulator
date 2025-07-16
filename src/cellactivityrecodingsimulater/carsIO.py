import json
import numpy as np
from pathlib import Path

from pydantic import BaseModel

from Site import Site
from Cell import Cell
from Settings import Settings

def load_settings(path: str) -> Settings:
    with open(path, "r") as f:
        return Settings(**json.load(f))
    
def load_cells(path: Path) -> list[Cell]:
    cells = []
    jcells = json.load(open(path, "r"))
    for i in range(len(jcells["id"])):
        cells.append(Cell(id=jcells["id"][i], 
                          x=jcells["x"][i], 
                          y=jcells["y"][i], 
                          z=jcells["z"][i]))
    return cells
    
def load_sites(path: Path) -> list[Site]:
    sites = []
    jsites = json.load(open(path, "r"))
    for i in range(len(jsites["id"])):
        sites.append(Site(id=jsites["id"][i], 
                          x=jsites["x"][i], 
                          y=jsites["y"][i],
                          z=jsites["z"][i]))
    return sites

def load_spikeTemplates(path: Path) -> list[np.ndarray]:
    spikeTemplates = []
    jspikeTemplates = json.load(open(path, "r"))
    for i in range(len(jspikeTemplates["id"])):
        spikeTemplates.append(np.array(jspikeTemplates["spikeTemplate"][i]))
    return spikeTemplates

def save_data(path: Path, cells: list[Cell], sites: list[Site]):

    # todo ファイル名を設定できるようにする
    # todo 必要事項全て保存できるようにする
    
    signalRaw = np.array([site.signalRaw for site in sites])
    signalNoise = np.array([site.signalNoise for site in sites])
    signalFiltered = np.array([site.signalFiltered for site in sites])
    
    # 異なる長さのリストをobject型で保存
    spikeTimeList = np.array([cell.spikeTimeList for cell in cells], dtype=object)
    spikeAmpList = np.array([cell.spikeAmpList for cell in cells], dtype=object)
    spikeTemp = np.array([cell.spikeTemp for cell in cells], dtype=object)

    # 浮動小数点データをint16に変換
    signalRaw_int16 = signalRaw.astype(np.int16)
    signalNoise_int16 = signalNoise.astype(np.int16)
    signalFiltered_int16 = signalFiltered.astype(np.int16)
    
    # int16データを.npyファイルとして保存
    np.save(path / "signalRaw_int16.npy", signalRaw_int16)
    np.save(path / "signalNoise_int16.npy", signalNoise_int16)
    np.save(path / "signalFiltered_int16.npy", signalFiltered_int16)
    
    # バイナリファイルに保存（int16）
    with open(path / "signalRaw.bin", "wb") as f:
        signalRaw_int16.flatten().tofile(f)
    with open(path / "signalNoise.bin", "wb") as f:
        signalNoise_int16.flatten().tofile(f)
    with open(path / "signalFiltered.bin", "wb") as f:
        signalFiltered_int16.flatten().tofile(f)

    cell_ids = np.array([cell.id for cell in cells])
    cell_positions = np.array([[cell.x, cell.y, cell.z] for cell in cells])
    spike_times = np.array([cell.spikeTimeList for cell in cells], dtype=object)
    spike_amps = np.array([cell.spikeAmpList for cell in cells], dtype=object)
    spike_temps = np.array([cell.spikeTemp for cell in cells], dtype=object)
    
    site_ids = np.array([site.id for site in sites])
    site_positions = np.array([[site.x, site.y, site.z] for site in sites])

    with open(path / "cell_ids.npy", "wb") as f:
        cell_ids.tofile(f)
    with open(path / "cell_positions.npy", "wb") as f:
        cell_positions.tofile(f)
    with open(path / "spike_times.npy", "wb") as f:
        spike_times.tofile(f)
    with open(path / "spike_amplitudes.npy", "wb") as f:
        spike_amps.tofile(f)
    with open(path / "spike_templates.npy", "wb") as f:
        spike_temps.tofile(f)

    with open(path / "site_ids.npy", "wb") as f:
        site_ids.tofile(f)
    with open(path / "site_positions.npy", "wb") as f:
        site_positions.tofile(f)

    # 互換性のために, 別ファイル名でも保存
    with open(path / "fire_times.npy", "wb") as f:
        spike_times.tofile(f)

    with open(path / "raw.bin", "wb") as f:
        signalRaw_int16.flatten().tofile(f)
    with open(path / "noise.bin", "wb") as f:
        signalNoise_int16.flatten().tofile(f)
    with open(path / "filt.bin", "wb") as f:
        signalFiltered_int16.flatten().tofile(f)

    


    