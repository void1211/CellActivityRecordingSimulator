from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    name: str
    pathSaveDir: Path | None

    fs: float
    duration: float # sec

    avgSpikeRate: float 
    isRefractory: bool
    refractoryPeriod: float # msec

    noiseType: str # "none", "normal", "gaussian", "truth"
    noiseAmp: float # uV
    pathTruthNoise: Path | None
    pathSitesOfTruthNoise: Path | None

    spikeType: str # "gabor", "truth", "template"
    pathSpikeList: Path | None
    isRandomSelect: bool
    gaborSigmaList: list[float] # msec
    gaborf0List: list[float] # Hz
    gaborthetaList: list[float] # rad
    spikeWidth: float # msec
    spikeAmpMax: float # uV
    spikeAmpMin: float # uV
    attenTime: float # msec
    
    # ノイズ細胞生成設定
    generate_noise_cells: bool = False
    cell_density: float = 30000  # cells/mm³
    margin: float = 100  # μm

    random_seed: int = 0  # 乱数シード値（デフォルト0）
    