from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Optional

class Settings(BaseModel):
    name: str
    pathSaveDir: Optional[Path] = None

    fs: float
    duration: float # sec

    avgSpikeRate: float 
    isRefractory: bool
    refractoryPeriod: float # msec
    absolute_refractory_ratio: float = 0.4  # 絶対不応期の割合（0.0-1.0）
    relative_refractory_prob: float = 0.3   # 相対不応期の発火確率（0.0-1.0）

    noiseType: str # "none", "normal", "gaussian", "truth", "model"
    noiseAmp: Optional[float] = None # uV
    pathTruthNoise: Optional[Path] = None
    pathSitesOfTruthNoise: Optional[Path] = None

    spikeType: str # "gabor", "truth", "template"
    pathSpikeList: Optional[Path] = None
    isRandomSelect: bool
    gaborSigmaList: Optional[list[float]] = None # msec
    gaborf0List: Optional[list[float]] = None # Hz
    gaborthetaList: Optional[list[float]] = None # rad
    spikeWidth: Optional[float] = None # msec
    spikeAmpMax: Optional[float] = None # uV
    spikeAmpMin: Optional[float] = None # uV
    attenTime: Optional[float] = None # msec
    
    # スパイクテンプレート類似度制御設定
    enable_template_similarity_control: bool = False  # テンプレート類似度制御を有効にするかどうか
    min_cosine_similarity: float = 0.7  # 最小コサイン類似度（-1.0-1.0）
    max_cosine_similarity: float = 0.95  # 最大コサイン類似度（-1.0-1.0）
    similarity_control_attempts: int = 100  # 類似度制御の最大試行回数
    
    # ノイズ細胞生成設定
    cell_density: float = 30000  # cells/mm³
    margin: float = 100  # μm

    random_seed: int = 0  # 乱数シード値（デフォルト0）

    # ドリフト設定
    enable_drift: bool = False  # ドリフトを有効にするかどうか
    drift_type: str = "linear"  # "linear", "exponential", "oscillatory", "random_walk", "step"
    drift_amplitude: float = 50.0  # μV
    drift_frequency: float = 0.1  # Hz (oscillatory drift用)
    common_drift: bool = True  # 電極全体で共通したドリフトを使用するかどうか

    # 電源ノイズ設定
    enable_power_noise: bool = False  # 電源ノイズを有効にするかどうか
    power_line_frequency: float = 50.0  # Hz (50Hz or 60Hz)
    power_noise_amplitude: float = 20.0  # μV
    common_power_noise: bool = True  # 電極全体で共通した電源ノイズを使用するかどうか

    @field_validator('noiseType')
    @classmethod
    def validate_noise_type(cls, v):
        valid_types = ["none", "normal", "gaussian", "truth", "model"]
        if v not in valid_types:
            raise ValueError(f"noiseType must be one of {valid_types}, got {v}")
        return v

    @field_validator('spikeType')
    @classmethod
    def validate_spike_type(cls, v):
        valid_types = ["gabor", "truth", "template"]
        if v not in valid_types:
            raise ValueError(f"spikeType must be one of {valid_types}, got {v}")
        return v

    @field_validator('drift_type')
    @classmethod
    def validate_drift_type(cls, v):
        valid_types = ["linear", "exponential", "oscillatory", "random_walk", "step"]
        if v not in valid_types:
            raise ValueError(f"drift_type must be one of {valid_types}, got {v}")
        return v

    @field_validator('min_cosine_similarity', 'max_cosine_similarity')
    @classmethod
    def validate_cosine_similarity(cls, v):
        if v < -1.0 or v > 1.0:
            raise ValueError(f"cosine similarity must be between -1.0 and 1.0, got {v}")
        return v

    def validate_settings(self) -> list[str]:
        """
        設定を検証し、エラーメッセージのリストを返す
        エラーがない場合は空のリストを返す
        """
        errors = []
        
        # 基本パラメータの検証
        if self.fs <= 0:
            errors.append("fs must be positive")
        if self.duration <= 0:
            errors.append("duration must be positive")
        if self.avgSpikeRate <= 0:
            errors.append("avgSpikeRate must be positive")
        if self.isRefractory and self.refractoryPeriod < 0:
            errors.append("refractoryPeriod must be non-negative when isRefractory is true")

        # noiseTypeに応じた検証
        if self.noiseType in ["normal", "gaussian"]:
            if self.noiseAmp is None:
                errors.append(f"noiseAmp is required when noiseType is {self.noiseType}")
            elif self.noiseAmp <= 0:
                errors.append("noiseAmp must be positive")
        
        elif self.noiseType == "truth":
            if self.pathTruthNoise is None:
                errors.append("pathTruthNoise is required when noiseType is 'truth'")
            elif not self.pathTruthNoise.exists():
                errors.append(f"pathTruthNoise file does not exist: {self.pathTruthNoise}")
        
        elif self.noiseType == "model":
            if self.cell_density <= 0:
                errors.append("cell_density must be positive when noiseType is 'model'")
            if self.margin < 0:
                errors.append("margin must be non-negative")

        # spikeTypeに応じた検証
        if self.spikeType == "gabor":
            if self.gaborSigmaList is None or len(self.gaborSigmaList) == 0:
                errors.append("gaborSigmaList is required when spikeType is 'gabor'")
            if self.gaborf0List is None or len(self.gaborf0List) == 0:
                errors.append("gaborf0List is required when spikeType is 'gabor'")
            if self.gaborthetaList is None or len(self.gaborthetaList) == 0:
                errors.append("gaborthetaList is required when spikeType is 'gabor'")
            if self.spikeWidth is None or self.spikeWidth <= 0:
                errors.append("spikeWidth must be positive when spikeType is 'gabor'")
        
        elif self.spikeType == "template":
            if self.pathSpikeList is None:
                errors.append("pathSpikeList is required when spikeType is 'template'")
            elif not self.pathSpikeList.exists():
                errors.append(f"pathSpikeList file does not exist: {self.pathSpikeList}")

        # 共通パラメータの検証
        if self.spikeAmpMax is not None and self.spikeAmpMin is not None:
            if self.spikeAmpMax <= self.spikeAmpMin:
                errors.append("spikeAmpMax must be greater than spikeAmpMin")
        
        if self.attenTime is not None and self.attenTime <= 0:
            errors.append("attenTime must be positive")

        # 類似度制御設定の検証
        if self.enable_template_similarity_control:
            if self.min_cosine_similarity >= self.max_cosine_similarity:
                errors.append("min_cosine_similarity must be less than max_cosine_similarity")
            if self.similarity_control_attempts <= 0:
                errors.append("similarity_control_attempts must be positive")

        return errors

    def is_valid(self) -> bool:
        """
        設定が有効かどうかを判定する
        """
        return len(self.validate_settings()) == 0

    def get_validation_summary(self) -> str:
        """
        検証結果のサマリーを返す
        """
        errors = self.validate_settings()
        if not errors:
            return "✓ 設定は有効です"
        else:
            return f"✗ 設定に{len(errors)}個のエラーがあります:\n" + "\n".join(f"  - {error}" for error in errors)
    