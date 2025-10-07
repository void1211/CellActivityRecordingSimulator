# from pydantic import BaseModel, field_validator
from pathlib import Path
import json
import logging
from .BaseSettings import BaseSettings
from .carsIO import load_cells_from_json, load_sites_from_json
class Settings(BaseSettings):
    def __init__(self, data: dict=None):
        super().__init__(data)
        self.rootSettings = RootSettings(safe_get(data, "baseSettings"))
        self.spikeSetting = SpikeSettings(safe_get(data, "spikeSettings"))
        self.noiseSettings = NoiseSettings(safe_get(data, "noiseSettings"))
        self.driftSettings = DriftSettings(safe_get(data, "driftSettings"))
        self.powerNoiseSettings = PowerNoiseSettings(safe_get(data, "powerNoiseSettings"))
        self.templateSimilarityControlSettings = TemplateSimilarityControlSettings(safe_get(data, "templateSimilarityControlSettings"))

    def validate(self) -> list[str]:
        """
        設定を検証し、エラーメッセージのリストを返す
        エラーがない場合は空のリストを返す
        """
        errors = []

        errors.extend(self.rootSettings.validate())
        errors.extend(self.spikeSetting.validate())
        errors.extend(self.noiseSettings.validate())
        errors.extend(self.driftSettings.validate())
        errors.extend(self.powerNoiseSettings.validate())
        errors.extend(self.templateSimilarityControlSettings.validate())
        return errors

    def is_valid(self) -> bool:
        """
        設定が有効かどうかを判定する
        """
        return len(self.validate()) == 0

    def get_validation_summary(self) -> str:
        """
        検証結果のサマリーを返す
        """
        errors = self.validate()
        if not errors:
            return "✓ 設定は有効です"
        else:
            return f"✗ 設定に{len(errors)}個のエラーがあります:\n" + "\n".join(f"  - {error}" for error in errors)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        data_dict = {
            "baseSettings": safe_get(data, "baseSettings"),
            "spikeSettings": safe_get(data, "spikeSettings"),
            "noiseSettings": safe_get(data, "noiseSettings"),
            "driftSettings": safe_get(data, "driftSettings"),
            "powerNoiseSettings": safe_get(data, "powerNoiseSettings"),
            "templateSimilarityControlSettings": safe_get(data, "templateSimilarityControlSettings"),
        }
        return cls(data_dict)
        

    def to_dict(self) -> dict:
        return {
            "baseSettings": self.rootSettings.to_dict(),
            "spikeSettings": self.spikeSetting.to_dict(),
            "noiseSettings": self.noiseSettings.to_dict(),
            "driftSettings": self.driftSettings.to_dict(),
            "powerNoiseSettings": self.powerNoiseSettings.to_dict(),
            "templateSimilarityControlSettings": self.templateSimilarityControlSettings.to_dict(),
        }

class RootSettings(BaseSettings):

    def __init__(self, data: dict):
        super().__init__(data)
        self.name = safe_get(data, "name")
        self.pathSaveDir = safe_get(data, "pathSaveDir")
        self.fs = safe_get(data, "fs")
        self.duration = safe_get(data, "duration")
        self.random_seed = safe_get(data, "random_seed")

    def validate(self) -> list[str]:
        errors = []
        if self.name is None:
            errors.append("name error.")
        if self.fs <= 0:
            errors.append("fs error.")
        if self.duration <= 0:
            errors.append("duration error.")
        if not isinstance(self.random_seed, int):
            errors.append("random_seed error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "RootSettings":
        data_dict = {
            "name": safe_get(data, "name"),
            "pathSaveDir": safe_get(data, "pathSaveDir"),
            "fs": safe_get(data, "fs"),
            "duration": safe_get(data, "duration"),
            "random_seed": safe_get(data, "random_seed"),
        }
        return cls(data_dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pathSaveDir": self.pathSaveDir,
            "fs": self.fs,
            "duration": self.duration,
            "random_seed": self.random_seed,
        }

class SpikeSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.avgSpikeRate = safe_get(data, "rate")
        self.isRefractory = safe_get(data, "isRefractory")
        self.refractoryPeriod = safe_get(data, "refractoryPeriod")
        self.absolute_refractory_ratio = safe_get(data, "absolute_refractory_ratio")
        self.spikeType = safe_get(data, "spikeType")
        self.amplitudeMax = safe_get(data, "amplitudeMax")
        self.amplitudeMin = safe_get(data, "amplitudeMin")
        self.attenTime = safe_get(data, "attenTime")
        if self.spikeType == "gabor":
            self.gabor = GaborSettings(safe_get(data, "gabor"))
        elif self.spikeType == "exponential":
            self.exponential = ExponentialSpikeSettings(safe_get(data, "exponential"))
        elif self.spikeType == "template":
            self.template = TemplateSettings(safe_get(data, "template"))
        elif self.spikeType == "truth":
            self.truth = TruthSpikeSettings(safe_get(data, "truth"))

    def validate(self) -> list[str]:
        errors = []
        if self.avgSpikeRate <= 0:
            errors.append("avgSpikeRate error.")
        if self.isRefractory and self.refractoryPeriod < 0:
            errors.append("refractoryPeriod error.")
        if self.absolute_refractory_ratio < 0 or self.absolute_refractory_ratio > 1:
            errors.append("absolute_refractory_ratio error.")
        if self.spikeType not in ["gabor", "exponential", "template", "truth"]:
            errors.append(f"spikeType error.")
        if self.amplitudeMax is None or self.amplitudeMax <= 0 or self.amplitudeMax <= self.amplitudeMin:
            errors.append("amplitudeMax error.")
        if self.amplitudeMin is None or self.amplitudeMin <= 0 or self.amplitudeMin >= self.amplitudeMax:
            errors.append("amplitudeMin error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "SpikeSettings":
        data_dict = {
            "rate": safe_get(data, "rate"),
            "isRefractory": safe_get(data, "isRefractory"),
            "refractoryPeriod": safe_get(data, "refractoryPeriod"),
            "absolute_refractory_ratio": safe_get(data, "absolute_refractory_ratio"),
            "spikeType": safe_get(data, "spikeType"),
            "amplitudeMax": safe_get(data, "amplitudeMax"),
            "amplitudeMin": safe_get(data, "amplitudeMin"),
            "attenTime": safe_get(data, "attenTime"),
        }
        if data_dict["spikeType"] == "gabor":
            data_dict["gabor"] = GaborSettings.from_dict(safe_get(data, "gabor"))
        elif data_dict["spikeType"] == "exponential":
            data_dict["exponential"] = ExponentialSpikeSettings.from_dict(safe_get(data, "exponential"))
        elif data_dict["spikeType"] == "template":
            data_dict["template"] = TemplateSettings.from_dict(safe_get(data, "template"))
        elif data_dict["spikeType"] == "truth":
            data_dict["truth"] = TruthSpikeSettings.from_dict(safe_get(data, "truth"))
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "rate": self.avgSpikeRate,
            "isRefractory": self.isRefractory,
            "refractoryPeriod": self.refractoryPeriod,
            "absolute_refractory_ratio": self.absolute_refractory_ratio,
            "spikeType": self.spikeType,
            "amplitudeMax": self.amplitudeMax,
            "amplitudeMin": self.amplitudeMin,
            "attenTime": self.attenTime,
        }
        if self.spikeType == "gabor":
            data_dict["gabor"] = self.gabor.to_dict()
        elif self.spikeType == "exponential":
            data_dict["exponential"] = self.exponential.to_dict()
        elif self.spikeType == "template":
            data_dict["template"] = self.template.to_dict()
        elif self.spikeType == "truth":
            data_dict["truth"] = self.truth.to_dict()
        return data_dict

class GaborSettings(BaseSettings):

    def __init__(self, data: dict):
        super().__init__(data)
        self.randType = safe_get(data, "randType")
        self.sigma = safe_get(data, "sigma")
        self.f0 = safe_get(data, "f0")
        self.theta = safe_get(data, "theta")
        self.width = safe_get(data, "width")

    def validate(self) -> list[str]:
        errors = []
        if self.randType not in ["list", "range"]:
            errors.append("randType error.")
        if self.randType == "list":
            if self.sigma is None or len(self.sigma) == 0:
                errors.append("sigma error.")
            if self.f0 is None or len(self.f0) == 0:
                errors.append("f0 error.")
            if self.theta is None or len(self.theta) == 0:
                errors.append("theta error.")
            if self.width is None or self.width <= 0:
                errors.append("width error.")
        if self.randType == "range":
            if self.sigma is None or len(self.sigma) > 2:
                errors.append("sigma error.")
            if self.f0 is None or len(self.f0) > 2:
                errors.append("f0 error.")
            if self.theta is None or len(self.theta) > 2:
                errors.append("theta error.")
            if self.width is None or self.width <= 0:
                errors.append("width error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "GaborSettings":
        data_dict = {
            "randType": safe_get(data, "randType"),
            "sigma": safe_get(data, "sigma"),
            "f0": safe_get(data, "f0"),
            "theta": safe_get(data, "theta"),
            "width": safe_get(data, "width"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "randType": self.randType,
            "sigma": self.sigma,
            "f0": self.f0,
            "theta": self.theta,
            "width": self.width,
        }
        return data_dict

class ExponentialSpikeSettings(BaseSettings):
    
    def __init__(self, data: dict):
        super().__init__(data)
        self.randType = safe_get(data, "randType")
        self.ms_before = safe_get(data, "ms_before")
        self.ms_after = safe_get(data, "ms_after")
        self.negative_amplitude = safe_get(data, "negative_amplitude")
        self.positive_amplitude = safe_get(data, "positive_amplitude")
        self.depolarization_ms = safe_get(data, "depolarization_ms")
        self.repolarization_ms = safe_get(data, "repolarization_ms")
        self.recovery_ms = safe_get(data, "recovery_ms")
        self.smooth_ms = safe_get(data, "smooth_ms")

    def validate(self) -> list[str]:
        errors = []
        if self.randType not in ["list", "range"]:
            errors.append("randType error.")
        if self.randType == "list":
            if self.ms_before is None or len(self.ms_before) == 0:
                errors.append("ms_before error.")
            if self.ms_after is None or len(self.ms_after) == 0:
                errors.append("ms_after error.")
            if self.negative_amplitude is None or len(self.negative_amplitude) == 0:
                errors.append("negative_amplitude error.")
            if self.positive_amplitude is None or len(self.positive_amplitude) == 0:
                errors.append("positive_amplitude error.")
            if self.depolarization_ms is None or len(self.depolarization_ms) == 0:
                errors.append("depolarization_ms error.")
            if self.repolarization_ms is None or len(self.repolarization_ms) == 0:
                errors.append("repolarization_ms error.")
            if self.recovery_ms is None or len(self.recovery_ms) == 0:
                errors.append("recovery_ms error.")
            if self.smooth_ms is None or len(self.smooth_ms) == 0:
                errors.append("smooth_ms error.")
        if self.randType == "range":
            if self.ms_before is None or len(self.ms_before) > 2:
                errors.append("ms_before error.")
            if self.ms_after is None or len(self.ms_after) > 2:
                errors.append("ms_after error.")
            if self.negative_amplitude is None or len(self.negative_amplitude) > 2:
                errors.append("negative_amplitude error.")
            if self.positive_amplitude is None or len(self.positive_amplitude) > 2:
                errors.append("positive_amplitude error.")
            if self.depolarization_ms is None or len(self.depolarization_ms) > 2:
                errors.append("depolarization_ms error.")
            if self.repolarization_ms is None or len(self.repolarization_ms) > 2:
                errors.append("repolarization_ms error.")
            if self.recovery_ms is None or len(self.recovery_ms) > 2:
                errors.append("recovery_ms error.")
            if self.smooth_ms is None or len(self.smooth_ms) > 2:
                errors.append("smooth_ms error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "ExponentialSpikeSettings":
        data_dict = {
            "randType": safe_get(data, "randType"),
            "ms_before": safe_get(data, "ms_before"),
            "ms_after": safe_get(data, "ms_after"),
            "negative_amplitude": safe_get(data, "negative_amplitude"),
            "positive_amplitude": safe_get(data, "positive_amplitude"),
            "depolarization_ms": safe_get(data, "depolarization_ms"),
            "repolarization_ms": safe_get(data, "repolarization_ms"),
            "recovery_ms": safe_get(data, "recovery_ms"),
            "smooth_ms": safe_get(data, "smooth_ms"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "randType": self.randType,
            "ms_before": self.ms_before,
            "ms_after": self.ms_after,
            "negative_amplitude": self.negative_amplitude,
            "positive_amplitude": self.positive_amplitude,
            "depolarization_ms": self.depolarization_ms,
            "repolarization_ms": self.repolarization_ms,
            "recovery_ms": self.recovery_ms,
            "smooth_ms": self.smooth_ms,
        }
        return data_dict

class TemplateSettings(BaseSettings):
    
    def __init__(self, data: dict):
        super().__init__(data)
        self.pathSpikeList = safe_get(data, "pathSpikeList")

    def validate(self) -> list[str]:
        errors = []
        if self.pathSpikeList is None:
            errors.append("pathSpikeList error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateSettings":
        data_dict = {
            "pathSpikeList": safe_get(data, "pathSpikeList"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "pathSpikeList": self.pathSpikeList,
        }
        return data_dict

class TruthSpikeSettings(BaseSettings):
    
    def __init__(self, data: dict):
        super().__init__(data)
        self.pathSpikeList = safe_get(data, "pathSpikeList")

    def validate(self) -> list[str]:
        errors = []
        if self.pathSpikeList is None:
            errors.append("pathSpikeList error.")
        return errors
    
    @classmethod
    def from_dict(cls, data: dict) -> "TruthSpikeSettings":
        data_dict = {
            "pathSpikeList": safe_get(data, "pathSpikeList"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "pathSpikeList": self.pathSpikeList,
        }
        return data_dict

class NoiseSettings(BaseSettings):

    def __init__(self, data: dict):
        super().__init__(data)
        self.noiseType = safe_get(data, "noiseType")
        self.model = ModelSettings(safe_get(data, "model"))
        self.normal = NormalSettings(safe_get(data, "normal"))
        self.gaussian = GaussianSettings(safe_get(data, "gaussian"))
        self.truth = TruthNoiseSettings(safe_get(data, "truth"))
    
    def validate(self) -> list[str]:
        errors = []
        if self.noiseType not in ["none", "normal", "gaussian", "truth", "model"]:
            errors.append(f"noiseType error: {self.noiseType}")
        return errors
    
    @classmethod
    def from_dict(cls, data: dict) -> "NoiseSettings":
        data_dict = {
            "noiseType": safe_get(data, "noiseType"),
            "model": ModelSettings.from_dict(safe_get(data, "model")),
            "normal": NormalSettings.from_dict(safe_get(data, "normal")),
            "gaussian": GaussianSettings.from_dict(safe_get(data, "gaussian")),
            "truth": TruthNoiseSettings.from_dict(safe_get(data, "truth")),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "noiseType": self.noiseType,
            "model": self.model.to_dict(),
            "normal": self.normal.to_dict(),
            "gaussian": self.gaussian.to_dict(),
            "truth": self.truth.to_dict(),
        }
        return data_dict
class NormalSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.noiseAmp = safe_get(data, "amplitude")
    def validate(self) -> list[str]:
        errors = []
        if self.noiseAmp is None or self.noiseAmp <= 0:
            errors.append("amplitude error.")
        return errors
    @classmethod
    def from_dict(cls, data: dict) -> "NormalSettings":
        data_dict = {
            "amplitude": safe_get(data, "amplitude"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "amplitude": self.noiseAmp,
        }
        return data_dict

class GaussianSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.noiseAmp = safe_get(data, "amplitude")
        self.noiseLoc = safe_get(data, "location")
        self.noiseScale = safe_get(data, "scale")
    def validate(self) -> list[str]:
        errors = []
        if self.noiseAmp is None or self.noiseAmp <= 0:
            errors.append("amplitude error.")
        if self.noiseLoc is None or self.noiseLoc < 0:
            errors.append("location error.")
        if self.noiseScale is None or self.noiseScale <= 0:
            errors.append("scale error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "GaussianSettings":
        data_dict = {
            "amplitude": safe_get(data, "amplitude"),
            "location": safe_get(data, "location"),
            "scale": safe_get(data, "scale"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "amplitude": self.noiseAmp,
            "location": self.noiseLoc,
            "scale": self.noiseScale,
        }
        return data_dict


class TruthNoiseSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.pathNoise = safe_get(data, "pathNoise")
        self.pathSites = safe_get(data, "pathSites")
    def validate(self) -> list[str]:
        errors = []
        if self.pathNoise is None:
            errors.append("pathNoise error.")
        if self.pathSites is None:
            errors.append("pathSites error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "TruthNoiseSettings":
        data_dict = {
            "pathNoise": safe_get(data, "pathNoise"),
            "pathSites": safe_get(data, "pathSites"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "pathNoise": self.pathNoise,
            "pathSites": self.pathSites,
        }
        return data_dict

class ModelSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.density = safe_get(data, "density")
        self.margin = safe_get(data, "margin")
        self.inviolableArea = safe_get(data, "inviolableArea")

    def validate(self) -> list[str]:
        errors = []
        if self.density <= 0:
            errors.append("density error.")
        if self.margin < 0:
            errors.append("margin error.")
        if self.inviolableArea < 0:
            errors.append("inviolableArea error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSettings":
        data_dict = {
            "density": safe_get(data, "density"),
            "margin": safe_get(data, "margin"),
            "inviolableArea": safe_get(data, "inviolableArea"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "density": self.density,
            "margin": self.margin,
            "inviolableArea": self.inviolableArea,
        }
        return data_dict

class DriftSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.enable = safe_get(data, "enable")
        self.driftType = safe_get(data, "driftType")
        self.randomWalk = RandomWalkSettings(safe_get(data, "random_walk"))
        self.step = StepSettings(safe_get(data, "step"))
        self.oscillatory = OscillatorySettings(safe_get(data, "oscillatory"))
        self.exponential = ExponentialDriftSettings(safe_get(data, "exponential"))

    def validate(self) -> list[str]:
        errors = []
        if self.enable and self.driftType not in ["random_walk", "step", "oscillatory", "exponential"]:
            errors.append(f"driftType error: {self.driftType}")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "DriftSettings":
        data_dict = {
            "enable": safe_get(data, "enable"),
            "driftType": safe_get(data, "driftType"),
        }
        if data_dict["driftType"] == "random_walk":
            data_dict["random_walk"] = RandomWalkSettings.from_dict(safe_get(data, "random_walk"))
        elif data_dict["driftType"] == "step":
            data_dict["step"] = StepSettings.from_dict(safe_get(data, "step"))
        elif data_dict["driftType"] == "oscillatory":
            data_dict["oscillatory"] = OscillatorySettings.from_dict(safe_get(data, "oscillatory"))
        elif data_dict["driftType"] == "exponential":
            data_dict["exponential"] = ExponentialDriftSettings.from_dict(safe_get(data, "exponential"))
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "enable": self.enable,
            "driftType": self.driftType,
        }
        if self.driftType == "random_walk":
            data_dict["random_walk"] = self.randomWalk.to_dict()
        elif self.driftType == "step":
            data_dict["step"] = self.step.to_dict()
        elif self.driftType == "oscillatory":
            data_dict["oscillatory"] = self.oscillatory.to_dict()
        elif self.driftType == "exponential":
            data_dict["exponential"] = self.exponential.to_dict()
        return data_dict

class RandomWalkSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = safe_get(data, "amplitude")
        self.frequency = safe_get(data, "frequency")
    
    def validate(self) -> list[str]:
        errors = []
        if self.noiseType not in ["none", "normal", "gaussian", "truth", "model"]:
            errors.append("noiseType error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "RandomWalkSettings":
        data_dict = {
            "amplitude": safe_get(data, "amplitude"),
            "frequency": safe_get(data, "frequency"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "amplitude": self.amplitude,
            "frequency": self.frequency,
        }
        return data_dict

class StepSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = safe_get(data, "amplitude")
        self.frequency = safe_get(data, "frequency")

    def validate(self) -> list[str]:
        errors = []
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        if self.frequency <= 0:
            errors.append("frequency error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "StepSettings":
        data_dict = {
            "amplitude": safe_get(data, "amplitude"),
            "frequency": safe_get(data, "frequency"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "amplitude": self.amplitude,
            "frequency": self.frequency,
        }
        return data_dict

class OscillatorySettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = safe_get(data, "amplitude")
        self.frequency = safe_get(data, "frequency")
    def validate(self) -> list[str]:
        errors = []
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        if self.frequency <= 0:
            errors.append("frequency error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "OscillatorySettings":
        data_dict = {
            "amplitude": safe_get(data, "amplitude"),
            "frequency": safe_get(data, "frequency"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "amplitude": self.amplitude,
            "frequency": self.frequency,
        }
        return data_dict
class ExponentialDriftSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = safe_get(data, "amplitude")
        self.frequency = safe_get(data, "frequency")
    def validate(self) -> list[str]:
        errors = []
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        if self.frequency <= 0:
            errors.append("frequency error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "ExponentialDriftSettings":
        data_dict = {
            "amplitude": safe_get(data, "amplitude"),
            "frequency": safe_get(data, "frequency"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "amplitude": self.amplitude,
            "frequency": self.frequency,
        }
        return data_dict

class PowerNoiseSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.enable = safe_get(data, "enable")
        self.frequency = safe_get(data, "frequency")
        self.amplitude = safe_get(data, "amplitude")
    def validate(self) -> list[str]:
        errors = []
        if self.enable and self.frequency <= 0:
            errors.append("frequency error.")
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "PowerNoiseSettings":
        data_dict = {
            "enable": safe_get(data, "enable"),
            "frequency": safe_get(data, "frequency"),
            "amplitude": safe_get(data, "amplitude"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "enable": self.enable,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
        }
        return data_dict

class TemplateSimilarityControlSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.enable = safe_get(data, "enable")
        self.min_cosine_similarity = safe_get(data, "min_cosine_similarity")
        self.max_cosine_similarity = safe_get(data, "max_cosine_similarity")
        self.similarity_control_attempts = safe_get(data, "similarity_control_attempts")
    def validate(self) -> list[str]:
        errors = []
        if self.enable and self.min_cosine_similarity >= self.max_cosine_similarity:
            errors.append("min_cosine_similarity must be less than max_cosine_similarity")
        if self.similarity_control_attempts <= 0:
            errors.append("similarity_control_attempts must be positive")
        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateSimilarityControlSettings":
        data_dict = {
            "enable": safe_get(data, "enable"),
            "min_cosine_similarity": safe_get(data, "min_cosine_similarity"),
            "max_cosine_similarity": safe_get(data, "max_cosine_similarity"),
            "similarity_control_attempts": safe_get(data, "similarity_control_attempts"),
        }
        return cls(data_dict)

    def to_dict(self) -> dict:
        data_dict = {
            "enable": self.enable,
            "min_cosine_similarity": self.min_cosine_similarity,
            "max_cosine_similarity": self.max_cosine_similarity,
            "similarity_control_attempts": self.similarity_control_attempts,
        }
        return data_dict

def default_settings(key: str=None) -> dict:
    default_settings ={
        "baseSettings":{
            "name": "default",
            "pathSaveDir": "default",
            "fs": 30000,
            "duration": 10,
            "random_seed": 0,
        },
        "spikeSettings":{
            "rate": 10,
            "isRefractory": True,
            "refractoryPeriod": 3,
            "absolute_refractory_ratio": 1.0,
            "amplitudeMax": 100,
            "amplitudeMin": 90,
            "attenTime": 25,
            "spikeType": "exponential",
            "gabor":{
                "randType": "list",
                "sigma": [0.4, 0.5, 0.6],
                "f0": [300, 400, 500],
                "theta": [0, 45, 90, 135, 180, 225, 270, 315],
                "width": 4,
            },
            "exponential":{
                "randType": "list",
                "ms_before": [4.0],
                "ms_after": [4.0],
                "negative_amplitude": [-1.0,-0.9],
                "positive_amplitude": [0.1, 0.3],
                "depolarization_ms": [0.1,0.3],
                "repolarization_ms": [0.4,1.0],
                "recovery_ms": [0.8,2.5],
                "smooth_ms": [0.05],
            },
            "template":{
                "pathSpikeList": "test_example_condition1.spike",
            },
            "truth":{
                "pathSpikeList": "test_example_condition1.spike",
            }
        },
        "noiseSettings":{
            "noiseType": "model",
            "model":{
                "density": 30000,
                "margin": 200,
                "inviolableArea": 0,
            },
            "normal":{
                "amplitude": 10,
            },
            "gaussian":{
                "amplitude": 10,
                "location": 0,
                "scale": 1,
            },
            "truth":{
                "pathNoise": "test_example_condition1.noise",
                "pathSites": "test_example_condition1.sites",
            },
        },
        "driftSettings":{
            "enable": True,
            "driftType": "random_walk",
            "random_walk":{
                "amplitude": 50.0,
                "frequency": 0.1,
            },
            "step":{
                "amplitude": 50.0,
                "frequency": 0.1,
            },
            "oscillatory":{
                "amplitude": 50.0,
                "frequency": 0.1,
            },
            "exponential":{
                "amplitude": 50.0,
                "frequency": 0.1,
            },
        },
        "powerNoiseSettings":{
            "enable": False,
            "frequency": 50.0,
            "amplitude": 20.0,
        },
        "templateSimilarityControlSettings":{
            "enable": False,
            "min_cosine_similarity": 0.7,
            "max_cosine_similarity": 0.95,
            "similarity_control_attempts": 100,
        },
    }
    return default_settings

def default_cells() -> dict:
    default_cells = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "x": [0, 0, 0, 0, 0, 0, 0, 0],
        "y": [0, 50, 100, 150, 200, 250, 300, 350],
        "z": [0, 0, 0, 0, 0, 0, 0, 0]
    }
    default_cells = load_cells_from_json(default_cells)
    return default_cells

def default_sites() -> dict:
    default_sites = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "x": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "y": [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375],
        "z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    default_sites = load_sites_from_json(default_sites)
    return default_sites

def safe_get(data: dict, key: str, default: any=None) -> any:
    try:
        return data[key]
    except KeyError:
        return {}

@classmethod
def convert_legacySettings(legacySettings: dict) -> dict:
    """レガシーな設定を新しい設定に変換する"""
    
    newSettings = {
        "baseSettings": {
            "name": safe_get(legacySettings, "name"),
            "pathSaveDir": safe_get(legacySettings, "pathSaveDir"),
            "fs": safe_get(legacySettings, "fs"),
            "duration": safe_get(legacySettings, "duration"),
            "random_seed": safe_get(legacySettings, "random_seed"),
        },
        "spikeSettings": {
            "rate": safe_get(legacySettings, "avgSpikeRate"),
            "isRefractory": safe_get(legacySettings, "isRefractory"),
            "refractoryPeriod": safe_get(legacySettings, "refractoryPeriod"),
            "absolute_refractory_ratio": safe_get(legacySettings, "absolute_refractory_ratio", 1.0),
            "amplitudeMax": safe_get(legacySettings, "spikeAmpMax"),
            "amplitudeMin": safe_get(legacySettings, "spikeAmpMin"),
            "attenTime": safe_get(legacySettings, "attenTime"),
            "spikeType": safe_get(legacySettings, "spikeType"),
            "gabor": {
                "randType": safe_get(legacySettings, "randType"),
                "sigma": safe_get(legacySettings, "sigma"),
                "f0": safe_get(legacySettings, "f0"),
                "theta": safe_get(legacySettings, "theta"),
                "width": safe_get(legacySettings, "spikeWidth"),
            },
            "exponential": {
                "randType": safe_get(legacySettings, "randType"),
                "ms_before": safe_get(legacySettings, "ms_before"),
                "ms_after": safe_get(legacySettings, "ms_after"),
                "negative_amplitude": safe_get(legacySettings, "negative_amplitude"),
                "positive_amplitude": safe_get(legacySettings, "positive_amplitude"),
                "depolarization_ms": safe_get(legacySettings, "depolarization_ms"),
                "repolarization_ms": safe_get(legacySettings, "repolarization_ms"),
                "recovery_ms": safe_get(legacySettings, "recovery_ms"),
                "smooth_ms": safe_get(legacySettings, "smooth_ms"),
            },
            "template": {
                "pathSpikeList": safe_get(legacySettings, "pathSpikeList"),
            },
            "truth": {
                "pathSpikeList": safe_get(legacySettings, "pathSpikeList"),
            },
        },
        "noiseSettings": {
            "noiseType": safe_get(legacySettings, "noiseType"),
            "model": {
                "density": safe_get(legacySettings, "density"),
                "margin": safe_get(legacySettings, "margin"),
                "inviolableArea": safe_get(legacySettings, "inviolableArea"),
            },
            "normal": {
                "amplitude": safe_get(legacySettings, "amplitude"),
            },
            "gaussian": {
                "amplitude": safe_get(legacySettings, "amplitude"),
                "location": safe_get(legacySettings, "loc"),
                "scale": safe_get(legacySettings, "scale"),
            },
            "truth": {
                "pathNoise": safe_get(legacySettings, "pathNoise"),
                "pathSites": safe_get(legacySettings, "pathSites"),
            },
        },
        "driftSettings": {
            "enable": safe_get(legacySettings, "enable_drift"),
            "driftType": safe_get(legacySettings, "drift_type"),
            "random_walk": {
                "amplitude": safe_get(legacySettings, "drift_amplitude"),
                "frequency": safe_get(legacySettings, "drift_frequency"),
            },
            "step": {
                "amplitude": safe_get(legacySettings, "drift_amplitude"),
                "frequency": safe_get(legacySettings, "drift_frequency"),
            },
            "oscillatory": {
                "amplitude": safe_get(legacySettings, "drift_amplitude"),
                "frequency": safe_get(legacySettings, "drift_frequency"),
            },
            "exponential": {
                "amplitude": safe_get(legacySettings, "drift_amplitude"),
                "frequency": safe_get(legacySettings, "drift_frequency"),
            },
        },
        "powerNoiseSettings": {
            "enable": safe_get(legacySettings, "enable_power_noise"),
            "frequency": safe_get(legacySettings, "power_line_frequency"),
            "amplitude": safe_get(legacySettings, "power_noise_amplitude"),
        },
        "templateSimilarityControlSettings": {
            "enable": safe_get(legacySettings, "enable_template_similarity_control"),
            "min_cosine_similarity": safe_get(legacySettings, "min_cosine_similarity"),
            "max_cosine_similarity": safe_get(legacySettings, "max_cosine_similarity"),
            "similarity_control_attempts": safe_get(legacySettings, "similarity_control_attempts"),
        },
    }

    return newSettings
    