#!/usr/bin/env python3
"""
kilosortのtemplates.npyファイルをロードしてテストするスクリプト
"""

import sys
from pathlib import Path
import numpy as np

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))

from templateMaker import TemplateMaker, load_kilosort_templates, analyze_kilosort_templates, filter_templates_by_distance

def test_template_maker(templates_path: str):
    """
    TemplateMakerクラスを使用してkilosortテンプレートをテストする
    
    Args:
        templates_path: templates.npyファイルのパス
    """
    try:
        # ファイルの存在確認
        path = Path(templates_path)
        if not path.exists():
            print(f"エラー: ファイルが見つかりません: {templates_path}")
            return
        
        print(f"templates.npyファイルをロード中: {path}")
        
        # TemplateMakerを使用してテンプレートをロード
        tm = TemplateMaker(path)
        
        print(f"\n=== 基本情報 ===")
        print(f"テンプレート形状: {tm.templates.shape}")
        print(f"クラスタ数: {tm.templates.shape[0]}")
        print(f"サンプル数: {tm.templates.shape[1]}")
        print(f"電極数: {tm.templates.shape[2]}")
        
        # 最大振幅を取得
        max_amplitudes = tm.get_max_amplitudes()
        print(f"\n=== 最大振幅 ===")
        print(f"最大振幅配列の形状: {max_amplitudes.shape}")
        print(f"データ型: {max_amplitudes.dtype}")
        print(f"最小値: {np.min(max_amplitudes):.6f}")
        print(f"最大値: {np.max(max_amplitudes):.6f}")
        print(f"平均値: {np.mean(max_amplitudes):.6f}")
        print(f"標準偏差: {np.std(max_amplitudes):.6f}")
        
        # 最初の数クラスタの最大振幅を表示
        print(f"\n=== 最初の5クラスタの最大振幅 ===")
        for i in range(min(5, max_amplitudes.shape[0])):
            cluster_max = np.max(max_amplitudes[i])
            cluster_min = np.min(max_amplitudes[i])
            print(f"クラスタ {i}: 最大={cluster_max:.6f}, 最小={cluster_min:.6f}")
        
        return tm
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def test_detailed_analysis(templates_path: str):
    """
    TemplateMakerの詳細分析機能をテストする
    
    Args:
        templates_path: templates.npyファイルのパス
    """
    try:
        path = Path(templates_path)
        if not path.exists():
            print(f"エラー: ファイルが見つかりません: {templates_path}")
            return
        
        print(f"\n=== 詳細分析 ===")
        tm = TemplateMaker(path)
        analysis = tm.analyze_templates()
        
        print(f"テンプレート形状: {analysis['shape']}")
        print(f"クラスタ数: {analysis['n_clusters']}")
        print(f"サンプル数: {analysis['n_samples']}")
        print(f"電極数: {analysis['n_electrodes']}")
        
        print(f"\n=== グローバル統計 ===")
        stats = analysis['global_stats']
        print(f"最大振幅: {stats['max_amplitude']:.6f}")
        print(f"最小振幅: {stats['min_amplitude']:.6f}")
        print(f"平均振幅: {stats['mean_amplitude']:.6f}")
        print(f"振幅標準偏差: {stats['std_amplitude']:.6f}")
        print(f"最大ピーク: {stats['max_peak']:.6f}")
        print(f"平均ピーク: {stats['mean_peak']:.6f}")
        print(f"ピーク標準偏差: {stats['std_peak']:.6f}")
        
        print(f"\n=== 最初の10クラスタの主要電極 ===")
        for i in range(min(10, analysis['n_clusters'])):
            primary_electrode = analysis['primary_electrodes'][i]
            peak_amp = analysis['peak_amplitudes'][i][primary_electrode]
            print(f"クラスタ {i}: 主要電極={primary_electrode}, ピーク振幅={peak_amp:.6f}")
        
        return analysis
        
    except Exception as e:
        print(f"詳細分析でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def test_filtering_functions(templates_path: str):
    """
    フィルタリング機能をテストする
    
    Args:
        templates_path: templates.npyファイルのパス
    """
    try:
        path = Path(templates_path)
        if not path.exists():
            print(f"エラー: ファイルが見つかりません: {templates_path}")
            return
        
        print(f"\n=== フィルタリング機能テスト ===")
        tm = TemplateMaker(path)
        
        # 距離計算のテスト
        print("距離行列計算中...")
        distances = tm.calculate_template_distances('euclidean')
        print(f"距離行列の形状: {distances.shape}")
        print(f"距離の範囲: {np.min(distances):.6f} - {np.max(distances):.6f}")
        
        # 距離フィルタリングのテスト
        print("\n距離フィルタリングテスト...")
        # 平均距離の中央値を閾値として使用
        avg_distances = np.mean(distances, axis=1)
        threshold = np.median(avg_distances)
        print(f"閾値: {threshold:.6f}")
        
        filtered_templates, selected_indices = tm.filter_by_distance_threshold(threshold, 'euclidean')
        print(f"フィルタリング結果: {len(selected_indices)}/{tm.templates.shape[0]} クラスタ")
        
        if len(selected_indices) > 0:
            # フィルタリングされたテンプレートで新しいTemplateMakerを作成
            filtered_tm = tm.create_filtered_template_maker(filtered_templates, selected_indices)
            print(f"フィルタリング後のテンプレート形状: {filtered_tm.templates.shape}")
            
            # フィルタリングされたテンプレートの分析
            filtered_analysis = filtered_tm.analyze_templates()
            print(f"フィルタリング後のクラスタ数: {filtered_analysis['n_clusters']}")
        
        # 振幅フィルタリングのテスト
        print("\n振幅フィルタリングテスト...")
        peak_amplitudes = tm.get_peak_amplitudes()
        max_peaks = np.max(peak_amplitudes, axis=1)
        min_threshold = np.percentile(max_peaks, 25)  # 25パーセンタイル
        max_threshold = np.percentile(max_peaks, 75)  # 75パーセンタイル
        print(f"振幅閾値: {min_threshold:.6f} - {max_threshold:.6f}")
        
        filtered_templates, selected_indices = tm.filter_by_amplitude_range(min_threshold, max_threshold)
        print(f"振幅フィルタリング結果: {len(selected_indices)}/{tm.templates.shape[0]} クラスタ")
        
        # 電極数フィルタリングのテスト
        print("\n電極数フィルタリングテスト...")
        filtered_templates, selected_indices = tm.filter_by_electrode_count(2, 10)
        print(f"電極数フィルタリング結果: {len(selected_indices)}/{tm.templates.shape[0]} クラスタ")
        
        # カスタム条件フィルタリングのテスト
        print("\nカスタム条件フィルタリングテスト...")
        def custom_condition(cluster_idx: int, tm: TemplateMaker, **kwargs) -> bool:
            # ピーク振幅が平均以上で、主要電極が特定の範囲内
            peak_amplitudes = tm.get_peak_amplitudes()
            primary_electrodes = tm.get_primary_electrodes()
            
            cluster_peak = np.max(peak_amplitudes[cluster_idx])
            primary_electrode = primary_electrodes[cluster_idx]
            
            mean_peak = np.mean(peak_amplitudes)
            
            return cluster_peak > mean_peak and primary_electrode < 5
        
        filtered_templates, selected_indices = tm.filter_templates_by_condition(custom_condition)
        print(f"カスタム条件フィルタリング結果: {len(selected_indices)}/{tm.templates.shape[0]} クラスタ")
        
    except Exception as e:
        print(f"フィルタリング機能でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def test_plotting(templates_path: str):
    """
    TemplateMakerのプロット機能をテストする
    
    Args:
        templates_path: templates.npyファイルのパス
    """
    try:
        path = Path(templates_path)
        if not path.exists():
            print(f"エラー: ファイルが見つかりません: {templates_path}")
            return
        
        print(f"\n=== プロット機能テスト ===")
        tm = TemplateMaker(path)
        
        # 最初のクラスタのテンプレートをプロット
        print("最初のクラスタのテンプレートをプロット中...")
        tm.plot_cluster_template(0)
        
        # 全クラスタの主要電極でのテンプレートをプロット
        print("全クラスタの主要電極でのテンプレートをプロット中...")
        tm.plot_all_clusters_primary_electrodes(max_clusters=10)
        
    except Exception as e:
        print(f"プロット機能でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def test_convenience_functions(templates_path: str):
    """
    便利な関数をテストする
    
    Args:
        templates_path: templates.npyファイルのパス
    """
    try:
        path = Path(templates_path)
        if not path.exists():
            print(f"エラー: ファイルが見つかりません: {templates_path}")
            return
        
        print(f"\n=== 便利な関数テスト ===")
        
        # load_kilosort_templates関数をテスト
        print("load_kilosort_templates関数をテスト中...")
        max_amplitudes = load_kilosort_templates(path)
        print(f"最大振幅配列の形状: {max_amplitudes.shape}")
        
        # analyze_kilosort_templates関数をテスト
        print("analyze_kilosort_templates関数をテスト中...")
        analysis = analyze_kilosort_templates(path)
        print(f"クラスタ数: {analysis['n_clusters']}")
        print(f"電極数: {analysis['n_electrodes']}")
        
        # filter_templates_by_distance関数をテスト
        print("filter_templates_by_distance関数をテスト中...")
        filtered_templates, selected_indices = filter_templates_by_distance(path, 2.0, 'euclidean')
        print(f"距離フィルタリング結果: {len(selected_indices)} クラスタ")
        
    except Exception as e:
        print(f"便利な関数でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def test_command_line_interface(templates_path: str):
    """
    コマンドラインインターフェースをテストする
    
    Args:
        templates_path: templates.npyファイルのパス
    """
    try:
        print(f"\n=== コマンドラインインターフェーステスト ===")
        
        # 基本的な実行
        print("基本的な実行:")
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "src/templateMaker/main.py", 
            templates_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 基本的な実行成功")
            print(result.stdout)
        else:
            print("✗ 基本的な実行失敗")
            print(result.stderr)
        
        # 詳細分析付きで実行
        print("\n詳細分析付きで実行:")
        result = subprocess.run([
            sys.executable, 
            "src/templateMaker/main.py", 
            templates_path,
            "--analyze"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 詳細分析実行成功")
            print(result.stdout)
        else:
            print("✗ 詳細分析実行失敗")
            print(result.stderr)
        
        # フィルタリング付きで実行
        print("\nフィルタリング付きで実行:")
        result = subprocess.run([
            sys.executable, 
            "src/templateMaker/main.py", 
            templates_path,
            "--filter-distance", "2.0",
            "--analyze"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ フィルタリング実行成功")
            print(result.stdout)
        else:
            print("✗ フィルタリング実行失敗")
            print(result.stderr)
        
    except Exception as e:
        print(f"コマンドラインインターフェースでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python test_kilosort_templates.py <templates.npyファイルのパス>")
        print("例: python test_kilosort_templates.py /path/to/templates.npy")
        sys.exit(1)
    
    templates_path = sys.argv[1]
    
    # 基本的なテスト
    tm = test_template_maker(templates_path)
    
    # 詳細分析
    analysis = test_detailed_analysis(templates_path)
    
    # フィルタリング機能テスト
    test_filtering_functions(templates_path)
    
    # 便利な関数テスト
    test_convenience_functions(templates_path)
    
    # コマンドラインインターフェーステスト
    test_command_line_interface(templates_path)
    
    # プロット機能テスト（オプション）
    try:
        import matplotlib
        test_plotting(templates_path)
    except ImportError:
        print("matplotlibがインストールされていないため、プロット機能はスキップします") 
