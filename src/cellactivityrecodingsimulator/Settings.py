# from pydantic import BaseModel, field_validator
from pathlib import Path
from typing import Optional
from enum import Enum
import json
import logging
from .BaseSettings import BaseSettings

"""
{
    "baseSettings":{
        "name": "test_example_condition1",
        "pathSaveDir": null,

        "fs": 30000,
        "duration": 10,
        "random_seed": 0,
    },
    "spikeSettings":{
        "rate": 10,
        "isRefractory": true,
        "refractoryPeriod": 3,
        "absolute_refractory_ratio": 1.0,
        "amplitudeMax": 100,
        "amplitudeMin": 90,
        "spikeType": "gabor",
            "gabor":{
            "randType": "list",
            "sigma": [0.4, 0.5, 0.6],
            "f0": [300, 400, 500],
            "theta": [0, 45, 90, 135, 180, 225, 270, 315],
            "width": 4,
        },
        "exponential":{
            "randType": "list",
            "ms_before": [0.4, 0.5, 0.6],
            "ms_after": [300, 400, 500],
            "negative_amplitude": [0, 45, 90, 135, 180, 225, 270, 315],
            "positive_amplitude": [0, 45, 90, 135, 180, 225, 270, 315],
            "depolarization_ms": [0, 45, 90, 135, 180, 225, 270, 315],
            "repolarization_ms": [0, 45, 90, 135, 180, 225, 270, 315],
            "recovery_ms": [0, 45, 90, 135, 180, 225, 270, 315],
            "smooth_ms": [0, 45, 90, 135, 180, 225, 270, 315],
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
            "margin": 100,
            "inviolableArea": 50,
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
        "enable": true,
        "driftType": "random_walk",
        "randomWalk":{
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
        "enable": false,
        "frequency": 50.0,
        "amplitude": 20.0,
    },
    "templateSimilarityControlSettings":{
        "enable": false,
        "min_cosine_similarity": 0.7,
        "max_cosine_similarity": 0.95,
        "similarity_control_attempts": 100,
    },
}
"""
# class SpikeType(Enum):
#     GABOR = "gabor"
#     TRUTH = "truth"
#     TEMPLATE = "template"
#     EXPONENTIAL = "exponential"

# class NoiseType(Enum):
#     NONE = "none"
#     NORMAL = "normal"
#     GAUSSIAN = "gaussian"
#     TRUTH = "truth"
#     MODEL = "model"

# class RandType(Enum):
#     LIST = "list"
#     RANGE = "range"

# class DriftType(Enum):
#     LINEAR = "linear"
#     EXPONENTIAL = "exponential"
#     OSCILLATORY = "oscillatory"
#     RANDOM_WALK = "random_walk"
#     STEP = "step"

class GaborSettings(BaseSettings):

    def __init__(self, data: dict):
        super().__init__(data)
        self.randType = data["randType"]
        self.sigma = data["sigma"]
        self.f0 = data["f0"]
        self.theta = data["theta"]
        self.width = data["width"]

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

class ExponentialSpikeSettings(BaseSettings):
    
    def __init__(self, data: dict):
        super().__init__(data)
        self.randType = data["randType"]
        self.ms_before = data["ms_before"]
        self.ms_after = data["ms_after"]
        self.negative_amplitude = data["negative_amplitude"]
        self.positive_amplitude = data["positive_amplitude"]
        self.depolarization_ms = data["depolarization_ms"]
        self.repolarization_ms = data["repolarization_ms"]
        self.recovery_ms = data["recovery_ms"]
        self.smooth_ms = data["smooth_ms"]

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

class TemplateSettings(BaseSettings):
    
    def __init__(self, data: dict):
        super().__init__(data)
        self.pathSpikeList = data["pathSpikeList"]

    def validate(self) -> list[str]:
        errors = []
        if self.pathSpikeList is None:
            errors.append("pathSpikeList error.")
        return errors

class TruthSpikeSettings(BaseSettings):
    
    def __init__(self, data: dict):
        super().__init__(data)
        self.pathSpikeList = data["pathSpikeList"]

    def validate(self) -> list[str]:
        errors = []
        if self.pathSpikeList is None:
            errors.append("pathSpikeList error.")
        return errors

class RootSettings(BaseSettings):

    def __init__(self, data: dict):
        super().__init__(data)
        self.name = data["name"]
        self.pathSaveDir = data["pathSaveDir"]
        self.fs = data["fs"]
        self.duration = data["duration"]
        self.random_seed = data["random_seed"]

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

class SpikeSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.avgSpikeRate = data["rate"]
        self.isRefractory = data["isRefractory"]
        self.refractoryPeriod = data["refractoryPeriod"]
        self.absolute_refractory_ratio = data["absolute_refractory_ratio"]
        self.spikeType = data["spikeType"]
        self.amplitudeMax = data["amplitudeMax"]
        self.amplitudeMin = data["amplitudeMin"]
        self.attenTime = data["attenTime"]
        if self.spikeType == "gabor":
            self.gabor = GaborSettings(data["gabor"])
        elif self.spikeType == "exponential":
            self.exponential = ExponentialSpikeSettings(data["exponential"])
        elif self.spikeType == "template":
            self.template = TemplateSettings(data["template"])
        elif self.spikeType == "truth":
            self.truth = TruthSpikeSettings(data["truth"])

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
class ModelSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.density = data["density"]
        self.margin = data["margin"]
        self.inviolableArea = data["inviolableArea"]

    def validate(self) -> list[str]:
        errors = []
        if self.density <= 0:
            errors.append("density error.")
        if self.margin < 0:
            errors.append("margin error.")
        if self.inviolableArea < 0:
            errors.append("inviolableArea error.")
        return errors

class NormalSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.noiseAmp = data["amplitude"]
    def validate(self) -> list[str]:
        errors = []
        if self.noiseAmp is None or self.noiseAmp <= 0:
            errors.append("amplitude error.")
        return errors

class GaussianSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.noiseAmp = data["amplitude"]
        self.noiseLoc = data["location"]
        self.noiseScale = data["scale"]
    def validate(self) -> list[str]:
        errors = []
        if self.noiseAmp is None or self.noiseAmp <= 0:
            errors.append("amplitude error.")
        if self.noiseLoc is None or self.noiseLoc < 0:
            errors.append("location error.")
        if self.noiseScale is None or self.noiseScale <= 0:
            errors.append("scale error.")
        return errors

class TruthNoiseSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.pathNoise = data["pathNoise"]
        self.pathSites = data["pathSites"]
    def validate(self) -> list[str]:
        errors = []
        if self.pathNoise is None:
            errors.append("pathNoise error.")
        if self.pathSites is None:
            errors.append("pathSites error.")
        return errors

class NoiseSettings(BaseSettings):

    def __init__(self, data: dict):
        super().__init__(data)
        self.noiseType = data["noiseType"]
        self.model = ModelSettings(data["model"])
        self.normal = NormalSettings(data["normal"])
        self.gaussian = GaussianSettings(data["gaussian"])
        self.truth = TruthNoiseSettings(data["truth"])
    
    def validate(self) -> list[str]:
        errors = []
        if self.noiseType not in ["none", "normal", "gaussian", "truth", "model"]:
            errors.append(f"noiseType error: {self.noiseType}")
        return errors

class RandomWalkSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = data["amplitude"]
        self.frequency = data["frequency"]

    def validate(self) -> list[str]:
        errors = []
        if self.noiseType not in ["none", "normal", "gaussian", "truth", "model"]:
            errors.append("noiseType error.")
        return errors

class StepSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = data["amplitude"]
        self.frequency = data["frequency"]

    def validate(self) -> list[str]:
        errors = []
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        if self.frequency <= 0:
            errors.append("frequency error.")
        return errors
class OscillatorySettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = data["amplitude"]
        self.frequency = data["frequency"]
    def validate(self) -> list[str]:
        errors = []
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        if self.frequency <= 0:
            errors.append("frequency error.")
        return errors

class ExponentialDriftSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.amplitude = data["amplitude"]
        self.frequency = data["frequency"]
    def validate(self) -> list[str]:
        errors = []
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        if self.frequency <= 0:
            errors.append("frequency error.")
        return errors

class DriftSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.enable = data["enable"]
        self.driftType = data["driftType"]
        if self.driftType == "random_walk": 
            self.randomWalk = RandomWalkSettings(data["random_walk"])
        elif self.driftType == "step":
            self.step = StepSettings(data["step"])
        elif self.driftType == "oscillatory":
            self.oscillatory = OscillatorySettings(data["oscillatory"])
        elif self.driftType == "exponential":
            self.exponential = ExponentialDriftSettings(data["exponential"])

    def validate(self) -> list[str]:
        errors = []
        if self.enable and self.driftType not in ["random_walk", "step", "oscillatory", "exponential"]:
            errors.append(f"driftType error: {self.driftType}")
        return errors

class PowerNoiseSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.enable = data["enable"]
        self.frequency = data["frequency"]
        self.amplitude = data["amplitude"]
    def validate(self) -> list[str]:
        errors = []
        if self.enable and self.frequency <= 0:
            errors.append("frequency error.")
        if self.amplitude <= 0:
            errors.append("amplitude error.")
        return errors

class TemplateSimilarityControlSettings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.enable = data["enable"]
        self.min_cosine_similarity = data["min_cosine_similarity"]
        self.max_cosine_similarity = data["max_cosine_similarity"]
        self.similarity_control_attempts = data["similarity_control_attempts"]
    def validate(self) -> list[str]:
        errors = []
        if self.enable and self.min_cosine_similarity >= self.max_cosine_similarity:
            errors.append("min_cosine_similarity must be less than max_cosine_similarity")
        if self.similarity_control_attempts <= 0:
            errors.append("similarity_control_attempts must be positive")
        return errors

# class DriftSettings(BaseSettings):
#     def __init__(self, data: dict):
#         super().__init__(data)
#         self.enable = data["enable"]
#         self.driftType = data["driftType"]
#         self.randomWalk = RandomWalkSettings(data["random_walk"])
#         self.step = StepSettings(data["step"])
#         self.oscillatory = OscillatorySettings(data["oscillatory"])
#         self.exponential = ExponentialSettings(data["exponential"])
#     def validate(self) -> list[str]:
#         errors = []
#         if self.enable and self.driftType not in ["random_walk", "step", "oscillatory", "exponential"]:
#             errors.append("driftType error.")
#         return errors

class Settings(BaseSettings):
    def __init__(self, data: dict):
        super().__init__(data)
        self.data = data
        self.rootSettings = RootSettings(data["baseSettings"])
        self.spikeSetting = SpikeSettings(data["spikeSettings"])
        self.noiseSettings = NoiseSettings(data["noiseSettings"])
        self.driftSettings = DriftSettings(data["driftSettings"])
        self.powerNoiseSettings = PowerNoiseSettings(data["powerNoiseSettings"])
        self.templateSimilarityControlSettings = TemplateSimilarityControlSettings(data["templateSimilarityControlSettings"])

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

    # def to_dict(self) -> dict:
    #     return self.data

    def from_json(self, json_path: Path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return self.to_dict(data) 
    