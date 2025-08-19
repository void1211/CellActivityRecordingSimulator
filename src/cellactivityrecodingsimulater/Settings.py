from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Optional

class Settings(BaseModel):
    name: str
    pathSaveDir: Optional[Path] = None

    fs: float
    duration: float # sec

    random_seed: int = 0  # 乱数シード値（デフォルト0）

    avgSpikeRate: float 
    isRefractory: bool
    refractoryPeriod: float # msec
    absolute_refractory_ratio: float = 0.4

    noiseType: str # "none", "normal", "gaussian", "truth", "model"
    noiseAmp: Optional[float] = None # uV
    pathTruthNoise: Optional[Path] = None
    pathSitesOfTruthNoise: Optional[Path] = None
    # ノイズ細胞生成設定
    density: float = 30000  # cells/mm³
    margin: float = 100  # μm


    spikeType: str # "gabor", "truth", "template", "expoential"
    # truth
    pathSpikeList: Optional[Path] = None
    isRandomSelect: bool = False

    # gabor
    randType: str = "list" # "list", "range"
    gaborSigma: Optional[list[float]] = None # msec
    gaborf0: Optional[list[float]] = None # Hz
    gabortheta: Optional[list[float]] = None # rad
    spikeWidth: Optional[float] = None # msec
    spikeAmpMax: Optional[float] = None # uV
    spikeAmpMin: Optional[float] = None # uV
    attenTime: Optional[float] = None # msec
    
    # exponential
    randType: str = "list" # "list", "range"
    ms_before: Optional[list[float]] = None # msec
    ms_after: Optional[list[float]] = None # msec
    negative_amplitude: Optional[list[float]] = None # uV
    positive_amplitude: Optional[list[float]] = None # uV
    depolarization_ms: Optional[list[float]] = None # msec
    repolarization_ms: Optional[list[float]] = None # msec
    recovery_ms: Optional[list[float]] = None # msec
    smooth_ms: Optional[list[float]] = None # msec

    # スパイクテンプレート類似度制御設定
    enable_template_similarity_control: bool = False  # テンプレート類似度制御を有効にするかどうか
    min_cosine_similarity: float = 0.7  # 最小コサイン類似度（-1.0-1.0）
    max_cosine_similarity: float = 0.95  # 最大コサイン類似度（-1.0-1.0）
    similarity_control_attempts: int = 100  # 類似度制御の最大試行回数

    # ドリフト設定
    enable_drift: bool = False  # ドリフトを有効にするかどうか
    drift_type: str = "linear"  # "linear", "exponential", "oscillatory", "random_walk", "step"
    drift_amplitude: float = 50.0  # μV
    drift_frequency: float = 0.1  # Hz (oscillatory drift用)

    # 電源ノイズ設定
    enable_power_noise: bool = False  # 電源ノイズを有効にするかどうか
    power_line_frequency: float = 50.0  # Hz (50Hz or 60Hz)
    power_noise_amplitude: float = 20.0  # μV

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
        valid_types = ["gabor", "truth", "template", "exponential"]
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
            if self.density <= 0:
                errors.append("density must be positive when noiseType is 'model'")
            if self.margin < 0:
                errors.append("margin must be non-negative")

        # spikeTypeに応じた検証
        if self.spikeType == "gabor":
            if self.gaborSigma is None:
                errors.append("gaborSigma is required when spikeType is 'gabor'")
            if self.gaborf0 is None:
                errors.append("gaborf0 is required when spikeType is 'gabor'")
            if self.gabortheta is None:
                errors.append("gabortheta is required when spikeType is 'gabor'")
            if self.spikeWidth is None or self.spikeWidth <= 0:
                errors.append("spikeWidth must be positive when spikeType is 'gabor'")
        
        elif self.spikeType == "template":
            if self.pathSpikeList is None:
                errors.append("pathSpikeList is required when spikeType is 'template'")
            elif not self.pathSpikeList.exists():
                errors.append(f"pathSpikeList file does not exist: {self.pathSpikeList}")

        elif self.spikeType == "exponential":
            if self.ms_before is None or len(self.ms_before) == 0:
                errors.append("ms_before must be positive when spikeType is 'exponential'")
            if self.ms_after is None or len(self.ms_after) == 0:
                errors.append("ms_after must be positive when spikeType is 'exponential'")
            if self.negative_amplitude is None or len(self.negative_amplitude) == 0 :
                errors.append("negative_amplitude must be positive when spikeType is 'exponential'")
            if self.positive_amplitude is None or len(self.positive_amplitude) == 0:
                errors.append("positive_amplitude must be positive when spikeType is 'exponential'")
            if self.depolarization_ms is None or len(self.depolarization_ms) == 0:
                errors.append("depolarization_ms must be positive when spikeType is 'exponential'")
            if self.repolarization_ms is None or len(self.repolarization_ms) == 0:
                errors.append("repolarization_ms must be positive when spikeType is 'exponential'")
            if self.recovery_ms is None or len(self.recovery_ms) == 0:
                errors.append("recovery_ms must be positive when spikeType is 'exponential'")
            if self.smooth_ms is None or len(self.smooth_ms) == 0:
                errors.append("smooth_ms must be positive when spikeType is 'exponential'")

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
    