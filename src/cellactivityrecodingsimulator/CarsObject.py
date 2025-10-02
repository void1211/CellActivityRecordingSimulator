from typing import List
from .Settings import Settings
from .Cell import Cell
from .Site import Site


class CarsObject:
    def __init__(
    self, 
    settings: Settings, 
    cells: List[Cell], 
    sites: List[Site],
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
