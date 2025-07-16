#!/usr/bin/env python3
"""
Template Maker Config Runner Test

設定ファイルランナーのテストスクリプト
"""

import sys
from pathlib import Path
import numpy as np

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from templateMaker import TemplateMakerRunner, create_default_config, save_config
from templateMaker.config_schema import TemplateMakerConfig


def create_test_templates():
    """テスト用のテンプレートファイルを作成"""
    # テスト用のテンプレートデータを作成
    n_clusters = 20
    n_samples = 82
    n_electrodes = 10
    
    # ランダムなテンプレートを作成
    templates = np.random.randn(n_clusters, n_samples, n_electrodes) * 0.5
    
    # 一部のクラスタに大きな振幅を追加
    for i in range(5):
        templates[i, 40:50, i] += np.random.randn(10) * 2.0
    
    # 保存
    test_templates_path = Path("test_templates.npy")
    np.save(test_templates_path, templates)
    print(f"テストテンプレートを作成しました: {test_templates_path}")
    
    return test_templates_path


def create_test_config():
    """テスト用の設定ファイルを作成"""
    config = TemplateMakerConfig(
        input={
            "templates_path": "test_templates.npy"
        },
        filters={
            "distance": {
                "max_distance": 2.0,
                "method": "euclidean"
            },
            "amplitude": {
                "min_amplitude": 0.1,
                "max_amplitude": 5.0
            },
            "electrode": {
                "min_electrodes": 1,
                "max_electrodes": 8
            },
            "apply_order": ["distance", "amplitude", "electrode"]
        },
        analysis={
            "enabled": True,
            "save_analysis": "test_analysis_results.npz"
        },
        plot={
            "enabled": False,
            "max_clusters": 5
        },
        output={
            "save_filtered": "test_filtered_templates.npy",
            "verbose": True
        },
        metadata={
            "description": "テスト用設定ファイル",
            "version": "1.0.0"
        }
    )
    
    config_path = Path("test_config.json")
    save_config(config, config_path)
    print(f"テスト設定ファイルを作成しました: {config_path}")
    
    return config_path


def test_config_runner():
    """設定ファイルランナーのテスト"""
    print("=== TemplateMaker Config Runner Test ===")
    
    try:
        # テストファイルを作成
        templates_path = create_test_templates()
        config_path = create_test_config()
        
        # 設定ファイルランナーを実行
        print("\n設定ファイルランナーを実行中...")
        runner = TemplateMakerRunner(config_path)
        runner.run()
        
        # 結果を確認
        if runner.filtered_templates is not None:
            print(f"\nフィルタリング結果:")
            print(f"元のクラスタ数: {runner.template_maker.templates.shape[0]}")
            print(f"フィルタリング後のクラスタ数: {runner.filtered_templates.shape[0]}")
            print(f"選択されたインデックス: {runner.selected_indices}")
        
        print("\nテスト完了!")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # テストファイルを削除
        cleanup_files = [
            "test_templates.npy",
            "test_config.json", 
            "test_filtered_templates.npy",
            "test_analysis_results.npz"
        ]
        
        for file_path in cleanup_files:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"テストファイルを削除しました: {file_path}")


def test_default_config_creation():
    """デフォルト設定作成のテスト"""
    print("\n=== デフォルト設定作成テスト ===")
    
    try:
        # デフォルト設定を作成
        config = create_default_config()
        print("デフォルト設定を作成しました")
        
        # 設定内容を表示
        print(f"入力ファイル: {config.input}")
        print(f"フィルタリング順序: {config.filters.apply_order}")
        print(f"分析有効: {config.analysis.enabled}")
        print(f"プロット有効: {config.plot.enabled}")
        
        # 設定を保存
        config_path = Path("default_config.json")
        save_config(config, config_path)
        print(f"デフォルト設定を保存しました: {config_path}")
        
        # 設定を再読み込み
        from templateMaker.config_schema import load_config
        loaded_config = load_config(config_path)
        print("設定を再読み込みしました")
        
        # 比較
        if config.model_dump() == loaded_config.model_dump():
            print("設定の保存・読み込みが正常に動作しました")
        else:
            print("設定の保存・読み込みに問題があります")
        
        # クリーンアップ
        config_path.unlink()
        print("テストファイルを削除しました")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # デフォルト設定作成テスト
    test_default_config_creation()
    
    # 設定ファイルランナーテスト
    test_config_runner() 
