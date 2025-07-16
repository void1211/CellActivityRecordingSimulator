"""
Template Analyzer for Kilosort Templates

kilosortのテンプレートファイルを解析し、スパイクテンプレートを生成・分析するためのモジュール
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("matplotlibがインストールされていません。プロット機能は使用できません。")


class TemplateMaker:
    """kilosortのテンプレートファイルを処理するクラス"""
    
    def __init__(self, templates_path: Path, scale_to_max_one: bool = False):
        """
        TemplateMakerの初期化
        
        Args:
            templates_path: kilosortのtemplates.npyファイルのパス
            scale_to_max_one: テンプレートを最大値1にスケーリングするかどうか
        """
        self.templates_path = Path(templates_path)
        self.scale_to_max_one = scale_to_max_one
        
        # ロガーを最初に初期化
        self.logger = logging.getLogger(__name__)
        
        if not self.templates_path.exists():
            raise FileNotFoundError(f"テンプレートファイルが見つかりません: {self.templates_path}")
        
        # テンプレートをロード
        self.templates = np.load(self.templates_path)
        
        # スケーリング処理
        if self.scale_to_max_one:
            self._scale_templates_to_max_one()
        
        self.logger.info(f"テンプレートをロードしました: {self.templates.shape}")
        if self.scale_to_max_one:
            self.logger.info("テンプレートを最大値1にスケーリングしました")
    
    def _scale_templates_to_max_one(self):
        """テンプレートを最大値1にスケーリング"""
        # 全テンプレートの最大値を取得
        global_max = np.max(np.abs(self.templates))
        
        if global_max > 0:
            # 最大値で正規化
            self.templates = self.templates / global_max
            self.logger.info(f"テンプレートをスケーリングしました: 最大値 = {global_max:.6f} -> 1.0")
        else:
            self.logger.warning("テンプレートの最大値が0です。スケーリングをスキップします。")
    
    def get_max_amplitudes(self) -> np.ndarray:
        """
        各クラスタの各電極での最大振幅を取得
        
        Returns:
            np.ndarray: クラスタ数 × 電極数の2次元配列
        """
        return np.max(self.templates, axis=1)
    
    def get_min_amplitudes(self) -> np.ndarray:
        """
        各クラスタの各電極での最小振幅を取得
        
        Returns:
            np.ndarray: クラスタ数 × 電極数の2次元配列
        """
        return np.min(self.templates, axis=1)
    
    def get_peak_amplitudes(self) -> np.ndarray:
        """
        各クラスタの各電極でのピーク振幅（絶対値の最大）を取得
        
        Returns:
            np.ndarray: クラスタ数 × 電極数の2次元配列
        """
        return np.max(np.abs(self.templates), axis=1)
    
    def get_primary_electrodes(self) -> np.ndarray:
        """
        各クラスタの主要電極（最大ピーク振幅を持つ電極）を取得
        
        Returns:
            np.ndarray: クラスタ数の1次元配列
        """
        peak_amplitudes = self.get_peak_amplitudes()
        return np.argmax(peak_amplitudes, axis=1)
    
    def analyze_templates(self) -> Dict[str, Any]:
        """
        テンプレートの詳細分析を実行
        
        Returns:
            Dict[str, Any]: 分析結果の辞書
        """
        n_clusters, n_samples, n_electrodes = self.templates.shape
        
        # ピーク振幅を取得
        peak_amplitudes = self.get_peak_amplitudes()
        primary_electrodes = self.get_primary_electrodes()
        
        # 統計情報を計算
        global_stats = {
            'max_peak': np.max(peak_amplitudes),
            'min_peak': np.min(peak_amplitudes),
            'mean_peak': np.mean(peak_amplitudes),
            'std_peak': np.std(peak_amplitudes),
            'median_peak': np.median(peak_amplitudes)
        }
        
        # 各クラスタの統計情報
        cluster_stats = []
        for i in range(n_clusters):
            cluster_peaks = peak_amplitudes[i]
            cluster_stats.append({
                'cluster_id': i,
                'max_peak': np.max(cluster_peaks),
                'min_peak': np.min(cluster_peaks),
                'mean_peak': np.mean(cluster_peaks),
                'std_peak': np.std(cluster_peaks),
                'primary_electrode': primary_electrodes[i],
                'active_electrodes': np.sum(cluster_peaks > 0.1 * np.max(cluster_peaks))
            })
        
        return {
            'shape': self.templates.shape,
            'n_clusters': n_clusters,
            'n_electrodes': n_electrodes,
            'n_samples': n_samples,
            'max_amplitudes': self.get_max_amplitudes(),
            'min_amplitudes': self.get_min_amplitudes(),
            'peak_amplitudes': peak_amplitudes,
            'primary_electrodes': primary_electrodes,
            'global_stats': global_stats,
            'cluster_stats': cluster_stats,
            'scale_to_max_one': self.scale_to_max_one
        }
    
    def calculate_template_distances(self, method: str = 'euclidean') -> np.ndarray:
        """
        テンプレート間の距離行列を計算
        
        Args:
            method: 距離計算方法 ('euclidean', 'cosine', 'correlation')
            
        Returns:
            np.ndarray: クラスタ数 × クラスタ数の距離行列
        """
        n_clusters = self.templates.shape[0]
        distances = np.zeros((n_clusters, n_clusters))
        
        # 各テンプレートを1次元ベクトルに変換
        templates_flat = self.templates.reshape(n_clusters, -1)
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if method == 'euclidean':
                    distances[i, j] = np.linalg.norm(templates_flat[i] - templates_flat[j])
                elif method == 'cosine':
                    # コサイン距離 = 1 - コサイン類似度
                    dot_product = np.dot(templates_flat[i], templates_flat[j])
                    norm_i = np.linalg.norm(templates_flat[i])
                    norm_j = np.linalg.norm(templates_flat[j])
                    if norm_i > 0 and norm_j > 0:
                        cosine_similarity = dot_product / (norm_i * norm_j)
                        distances[i, j] = 1 - cosine_similarity
                    else:
                        distances[i, j] = 1.0
                elif method == 'correlation':
                    # 相関距離 = 1 - 相関係数
                    correlation = np.corrcoef(templates_flat[i], templates_flat[j])[0, 1]
                    if not np.isnan(correlation):
                        distances[i, j] = 1 - correlation
                    else:
                        distances[i, j] = 1.0
                else:
                    raise ValueError(f"未知の距離計算方法: {method}")
        
        return distances
    
    def filter_templates_by_condition(self, condition_func: Callable) -> Tuple[np.ndarray, List[int]]:
        """
        カスタム条件でテンプレートをフィルタリング
        
        Args:
            condition_func: 条件関数 (cluster_idx, tm) -> bool
            
        Returns:
            Tuple[np.ndarray, List[int]]: フィルタリングされたテンプレートと選択されたインデックス
        """
        selected_indices = []
        
        for i in range(self.templates.shape[0]):
            if condition_func(i, self):
                selected_indices.append(i)
        
        if selected_indices:
            filtered_templates = self.templates[selected_indices]
            return filtered_templates, selected_indices
        else:
            return np.array([]), []
    
    def filter_by_distance_threshold(self, threshold: float, method: str = 'euclidean', 
                                   reference_cluster: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        距離閾値でテンプレートをフィルタリング
        
        Args:
            threshold: 距離閾値
            method: 距離計算方法
            reference_cluster: 基準となるクラスタのインデックス（Noneの場合は全クラスタとの平均距離）
            
        Returns:
            Tuple[np.ndarray, List[int]]: フィルタリングされたテンプレートと選択されたインデックス
        """
        distances = self.calculate_template_distances(method)
        selected_indices = []
        
        for i in range(self.templates.shape[0]):
            if reference_cluster is not None:
                # 特定のクラスタとの距離をチェック
                distance = distances[i, reference_cluster]
            else:
                # 全クラスタとの平均距離をチェック
                distance = np.mean(distances[i, :])
            
            if distance <= threshold:
                selected_indices.append(i)
        
        if selected_indices:
            filtered_templates = self.templates[selected_indices]
            return filtered_templates, selected_indices
        else:
            return np.array([]), []
    
    def filter_by_amplitude_range(self, min_amplitude: float, max_amplitude: float) -> Tuple[np.ndarray, List[int]]:
        """
        振幅範囲でテンプレートをフィルタリング
        
        Args:
            min_amplitude: 最小振幅閾値
            max_amplitude: 最大振幅閾値
            
        Returns:
            Tuple[np.ndarray, List[int]]: フィルタリングされたテンプレートと選択されたインデックス
        """
        peak_amplitudes = self.get_peak_amplitudes()
        max_peaks = np.max(peak_amplitudes, axis=1)  # 各クラスタの最大ピーク振幅
        
        selected_indices = []
        for i in range(self.templates.shape[0]):
            if min_amplitude <= max_peaks[i] <= max_amplitude:
                selected_indices.append(i)
        
        if selected_indices:
            filtered_templates = self.templates[selected_indices]
            return filtered_templates, selected_indices
        else:
            return np.array([]), []
    
    def filter_by_electrode_count(self, min_electrodes: int, max_electrodes: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        電極数でテンプレートをフィルタリング
        
        Args:
            min_electrodes: 最小電極数
            max_electrodes: 最大電極数（Noneの場合は制限なし）
            
        Returns:
            Tuple[np.ndarray, List[int]]: フィルタリングされたテンプレートと選択されたインデックス
        """
        peak_amplitudes = self.get_peak_amplitudes()
        
        selected_indices = []
        for i in range(self.templates.shape[0]):
            # アクティブな電極数をカウント（ピーク振幅が最大値の10%以上）
            cluster_peaks = peak_amplitudes[i]
            threshold = np.max(cluster_peaks) * 0.1
            active_electrodes = np.sum(cluster_peaks >= threshold)
            
            if min_electrodes <= active_electrodes:
                if max_electrodes is None or active_electrodes <= max_electrodes:
                    selected_indices.append(i)
        
        if selected_indices:
            filtered_templates = self.templates[selected_indices]
            return filtered_templates, selected_indices
        else:
            return np.array([]), []
    
    def create_filtered_template_maker(self, filtered_templates: np.ndarray, selected_indices: List[int]) -> 'TemplateMaker':
        """
        フィルタリングされたテンプレートで新しいTemplateMakerインスタンスを作成
        
        Args:
            filtered_templates: フィルタリングされたテンプレート
            selected_indices: 選択されたインデックス
            
        Returns:
            TemplateMaker: 新しいTemplateMakerインスタンス
        """
        # 新しいインスタンスを作成
        new_tm = TemplateMaker.__new__(TemplateMaker)
        new_tm.templates = filtered_templates
        new_tm.templates_path = self.templates_path
        new_tm.scale_to_max_one = self.scale_to_max_one
        new_tm.logger = self.logger
        return new_tm
    
    def plot_cluster_template(self, cluster_idx: int, save_path: Optional[Path] = None):
        """
        指定されたクラスタのテンプレートをプロット
        
        Args:
            cluster_idx: クラスタのインデックス
            save_path: 保存先パス（オプション）
        """
        if not HAS_MATPLOTLIB:
            print("matplotlibがインストールされていません")
            return
        
        if cluster_idx >= self.templates.shape[0]:
            print(f"クラスタインデックス {cluster_idx} が範囲外です")
            return
        
        template = self.templates[cluster_idx]
        n_samples, n_electrodes = template.shape
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 各電極のテンプレートをプロット
        for electrode in range(n_electrodes):
            ax.plot(template[:, electrode], label=f'Electrode {electrode}')
        
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Template for Cluster {cluster_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"プロットを保存しました: {save_path}")
        
        plt.show()
    
    def plot_all_clusters_primary_electrodes(self, max_clusters: int = 10, save_path: Optional[Path] = None):
        """
        全クラスタの主要電極でのテンプレートをプロット
        
        Args:
            max_clusters: プロットする最大クラスタ数
            save_path: 保存先パス（オプション）
        """
        if not HAS_MATPLOTLIB:
            print("matplotlibがインストールされていません")
            return
        
        n_clusters = min(max_clusters, self.templates.shape[0])
        primary_electrodes = self.get_primary_electrodes()
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(n_clusters):
            if i < len(axes):
                template = self.templates[i]
                primary_electrode = primary_electrodes[i]
                
                # 主要電極のテンプレートをプロット
                axes[i].plot(template[:, primary_electrode], 'b-', linewidth=2)
                axes[i].set_title(f'Cluster {i} (Electrode {primary_electrode})')
                axes[i].set_xlabel('Sample')
                axes[i].set_ylabel('Amplitude')
                axes[i].grid(True, alpha=0.3)
        
        # 未使用のサブプロットを非表示
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"プロットを保存しました: {save_path}")
        
        plt.show()
    
    def save_analysis(self, save_path: Path):
        """
        分析結果を保存
        
        Args:
            save_path: 保存先パス
        """
        analysis = self.analyze_templates()
        
        # 保存可能なデータのみを抽出
        save_data = {
            'shape': analysis['shape'],
            'n_clusters': analysis['n_clusters'],
            'n_electrodes': analysis['n_electrodes'],
            'n_samples': analysis['n_samples'],
            'max_amplitudes': analysis['max_amplitudes'],
            'min_amplitudes': analysis['min_amplitudes'],
            'peak_amplitudes': analysis['peak_amplitudes'],
            'primary_electrodes': analysis['primary_electrodes'],
            'global_stats': analysis['global_stats'],
            'scale_to_max_one': analysis['scale_to_max_one']
        }
        
        np.savez(save_path, **save_data)
        print(f"分析結果を保存しました: {save_path}")


# 便利な関数
def load_kilosort_templates(templates_path: Path, scale_to_max_one: bool = False) -> np.ndarray:
    """
    kilosortのtemplates.npyファイルをロードして、各クラスタの最大振幅を取得
    
    Args:
        templates_path: templates.npyファイルのパス
        scale_to_max_one: テンプレートを最大値1にスケーリングするかどうか
        
    Returns:
        np.ndarray: クラスタ数 × 電極数の2次元配列
    """
    tm = TemplateMaker(templates_path, scale_to_max_one=scale_to_max_one)
    return tm.get_max_amplitudes()


def analyze_kilosort_templates(templates_path: Path, scale_to_max_one: bool = False) -> Dict[str, Any]:
    """
    kilosortのテンプレートファイルの詳細分析
    
    Args:
        templates_path: templates.npyファイルのパス
        scale_to_max_one: テンプレートを最大値1にスケーリングするかどうか
        
    Returns:
        Dict[str, Any]: 分析結果の辞書
    """
    tm = TemplateMaker(templates_path, scale_to_max_one=scale_to_max_one)
    return tm.analyze_templates()


def filter_templates_by_distance(templates_path: Path, threshold: float, method: str = 'euclidean',
                                reference_cluster: Optional[int] = None, scale_to_max_one: bool = False) -> Tuple[np.ndarray, List[int]]:
    """
    距離閾値でテンプレートをフィルタリングする便利な関数
    
    Args:
        templates_path: templates.npyファイルのパス
        threshold: 距離閾値
        method: 距離計算方法
        reference_cluster: 基準となるクラスタのインデックス
        scale_to_max_one: テンプレートを最大値1にスケーリングするかどうか
        
    Returns:
        Tuple[np.ndarray, List[int]]: フィルタリングされたテンプレート配列と選択されたインデックス
    """
    tm = TemplateMaker(templates_path, scale_to_max_one=scale_to_max_one)
    return tm.filter_by_distance_threshold(threshold, method, reference_cluster) 
