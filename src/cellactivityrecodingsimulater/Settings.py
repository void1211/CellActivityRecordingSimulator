from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    pathCell: Path
    pathSite: Path
    pathSaveDir: Path

    fs: float
    duration: float # sec

    avgSpikeRate: float 
    isRefractory: bool
    refractoryPeriod: float # msec

    noiseType: str # "none", "gaussian", "truth"
    noiseAmp: float # uV
    pathTruthNoise: Path
    pathSitesOfTruthNoise: Path

    spikeType: str # "gabor", "truth", "template"
    pathSpikeList: Path
    isRandomSelect: bool
    gaborSigmaList: list[float]
    gaborf0List: list[float]
    gaborthetaList: list[float]
    spikeWidth: float # msec
    spikeAmp: float # uV
    attenTime: float # msec

