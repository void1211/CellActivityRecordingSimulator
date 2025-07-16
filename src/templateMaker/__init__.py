"""
Template Maker Package

kilosortのテンプレートファイルを処理し、スパイクテンプレートを生成・分析するためのパッケージ
"""

from .template_analyzer import (
    TemplateMaker, 
    load_kilosort_templates, 
    analyze_kilosort_templates,
    filter_templates_by_distance
)

from .config_schema import (
    TemplateMakerConfig,
    create_default_config,
    save_config,
    load_config
)

from .config_runner import (
    TemplateMakerRunner,
    run_from_config
)

__all__ = [
    # テンプレート解析
    'TemplateMaker',
    'load_kilosort_templates', 
    'analyze_kilosort_templates',
    'filter_templates_by_distance',
    
    # 設定ファイル関連
    'TemplateMakerConfig',
    'create_default_config',
    'save_config',
    'load_config',
    
    # 設定ファイル実行
    'TemplateMakerRunner',
    'run_from_config'
]

__version__ = "1.0.0" 
