"""
Configuration Schema for Template Maker

設定ファイルのスキーマを定義するモジュール
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class InputConfig(BaseModel):
    """入力設定"""
    templates_path: str = Field(..., description="templates.npyファイルのパス")
    scale_to_max_one: bool = Field(False, description="テンプレートを最大値1にスケーリングするかどうか")


class DistanceFilterConfig(BaseModel):
    """距離フィルタリングの設定"""
    max_distance: Optional[float] = Field(None, description="最大距離閾値")
    min_distance: Optional[float] = Field(None, description="最小距離閾値")
    method: str = Field("euclidean", description="距離計算方法 (euclidean, cosine, correlation)")
    reference_cluster: Optional[int] = Field(None, description="基準となるクラスタのインデックス")


class AmplitudeFilterConfig(BaseModel):
    """振幅フィルタリングの設定"""
    min_amplitude: Optional[float] = Field(None, description="最小振幅閾値")
    max_amplitude: Optional[float] = Field(None, description="最大振幅閾値")


class ElectrodeFilterConfig(BaseModel):
    """電極数フィルタリングの設定"""
    min_electrodes: Optional[int] = Field(None, description="最小電極数")
    max_electrodes: Optional[int] = Field(None, description="最大電極数")


class CustomFilterConfig(BaseModel):
    """カスタムフィルタリングの設定"""
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="カスタム条件のリスト")


class FilterConfig(BaseModel):
    """フィルタリング設定の統合"""
    distance: Optional[DistanceFilterConfig] = Field(None, description="距離フィルタリング設定")
    amplitude: Optional[AmplitudeFilterConfig] = Field(None, description="振幅フィルタリング設定")
    electrode: Optional[ElectrodeFilterConfig] = Field(None, description="電極数フィルタリング設定")
    custom: Optional[CustomFilterConfig] = Field(None, description="カスタムフィルタリング設定")
    apply_order: List[str] = Field(default_factory=list, description="フィルタリング適用順序")


class AnalysisConfig(BaseModel):
    """分析設定"""
    enabled: bool = Field(True, description="詳細分析を実行するかどうか")
    save_analysis: Optional[str] = Field(None, description="分析結果の保存先パス")


class PlotConfig(BaseModel):
    """プロット設定"""
    enabled: bool = Field(False, description="プロットを実行するかどうか")
    max_clusters: int = Field(10, description="プロットする最大クラスタ数")
    save_plots: Optional[str] = Field(None, description="プロットの保存先ディレクトリ")


class OutputConfig(BaseModel):
    """出力設定"""
    save_filtered: Optional[str] = Field(None, description="フィルタリングされたテンプレートの保存先パス")
    save_analysis: Optional[str] = Field(None, description="分析結果の保存先パス")
    save_plots: Optional[str] = Field(None, description="プロットの保存先ディレクトリ")
    verbose: bool = Field(True, description="詳細な出力を表示するかどうか")


class TemplateMakerConfig(BaseModel):
    """TemplateMakerの設定ファイルスキーマ"""
    # 入力設定
    input: InputConfig = Field(..., description="入力ファイル設定")
    
    # フィルタリング設定
    filters: Optional[FilterConfig] = Field(None, description="フィルタリング設定")
    
    # 分析設定
    analysis: Optional[AnalysisConfig] = Field(None, description="分析設定")
    
    # プロット設定
    plot: Optional[PlotConfig] = Field(None, description="プロット設定")
    
    # 出力設定
    output: Optional[OutputConfig] = Field(None, description="出力設定")
    
    # メタデータ
    metadata: Optional[Dict[str, Any]] = Field(None, description="メタデータ")


def create_default_config() -> TemplateMakerConfig:
    """デフォルト設定を作成"""
    return TemplateMakerConfig(
        input=InputConfig(
            templates_path="templates.npy",
            scale_to_max_one=False
        ),
        filters=FilterConfig(
            distance=DistanceFilterConfig(
                max_distance=2.0,
                method="euclidean"
            ),
            amplitude=AmplitudeFilterConfig(
                min_amplitude=0.1,
                max_amplitude=10.0
            ),
            electrode=ElectrodeFilterConfig(
                min_electrodes=1,
                max_electrodes=10
            ),
            apply_order=["distance", "amplitude", "electrode"]
        ),
        analysis=AnalysisConfig(
            enabled=True,
            save_analysis="analysis_results.npz"
        ),
        plot=PlotConfig(
            enabled=False,
            max_clusters=10
        ),
        output=OutputConfig(
            save_filtered="filtered_templates.npy",
            verbose=True
        ),
        metadata={
            "description": "TemplateMaker設定ファイル",
            "version": "1.0.0",
            "created_by": "TemplateMaker"
        }
    )


def save_config(config: TemplateMakerConfig, path: Path) -> None:
    """設定をJSONファイルに保存"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(config.model_dump_json(indent=2, ensure_ascii=False))


def load_config(path: Path) -> TemplateMakerConfig:
    """JSONファイルから設定を読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return TemplateMakerConfig(**config_dict) 
