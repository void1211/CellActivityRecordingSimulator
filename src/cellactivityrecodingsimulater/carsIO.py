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
    signalRaw = np.array([site.signalRaw for site in sites])
    signalNoise = np.array([site.signalNoise for site in sites])
    
    np.save(path / "signalRaw.npy", signalRaw)
    np.save(path / "signalNoise.npy", signalNoise)

    