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

    np.save(path / "signalRaw.npy", signalRaw)
    np.save(path / "signalNoise.npy", signalNoise)
    np.save(path / "signalFiltered.npy", signalFiltered)
    np.save(path / "spikeTimeList.npy", spikeTimeList)
    np.save(path / "spikeAmpList.npy", spikeAmpList)
    np.save(path / "spikeTemp.npy", spikeTemp)
    
    # バイナリファイルに保存
    with open(path / "signalRaw.bin", "wb") as f:
        signalRaw.flatten().tofile(f)
    with open(path / "signalNoise.bin", "wb") as f:
        signalNoise.flatten().tofile(f)
    with open(path / "signalFiltered.bin", "wb") as f:
        signalFiltered.flatten().tofile(f)
    