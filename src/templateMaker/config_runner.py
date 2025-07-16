"""
Configuration Runner for Template Maker

設定ファイルを実行するためのランナーモジュール
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .template_analyzer import TemplateMaker
from .config_schema import TemplateMakerConfig, load_config


class TemplateMakerRunner:
    """設定ファイルを実行するランナークラス"""
    
    def __init__(self, config_path: Path):
        """
        TemplateMakerRunnerの初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.template_maker = None
        self.filtered_templates = None
        self.selected_indices = None
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO if self.config.output.verbose else logging.WARNING,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> None:
        """設定ファイルに基づいて処理を実行"""
        try:
            self.logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            
            # テンプレートをロード
            self._load_templates()
            
            # フィルタリングを実行
            if self.config.filters:
                self._apply_filters()
            
            # 分析を実行
            if self.config.analysis and self.config.analysis.enabled:
                self._run_analysis()
            
            # プロットを実行
            if self.config.plot and self.config.plot.enabled:
                self._run_plots()
            
            # 結果を保存
            self._save_results()
            
            self.logger.info("処理が完了しました")
            
        except Exception as e:
            self.logger.error(f"処理中にエラーが発生しました: {e}")
            raise
    
    def _load_templates(self) -> None:
        """テンプレートをロード"""
        templates_path = Path(self.config.input.templates_path)
        
        if not templates_path.exists():
            raise FileNotFoundError(f"テンプレートファイルが見つかりません: {templates_path}")
        
        # スケーリングオプションを適用
        scale_to_max_one = self.config.input.scale_to_max_one
        self.template_maker = TemplateMaker(templates_path, scale_to_max_one=scale_to_max_one)
        self.logger.info(f"テンプレートをロードしました: {self.template_maker.templates.shape}")
        if scale_to_max_one:
            self.logger.info("テンプレートを最大値1にスケーリングしました")
    
    def _apply_filters(self) -> None:
        """フィルタリングを適用"""
        self.logger.info("フィルタリングを開始します")
        
        current_tm = self.template_maker
        current_indices = list(range(current_tm.templates.shape[0]))
        
        # 設定された順序でフィルタリングを適用
        for filter_type in self.config.filters.apply_order:
            if filter_type == "distance" and self.config.filters.distance:
                current_tm, current_indices = self._apply_distance_filter(current_tm, current_indices)
            elif filter_type == "amplitude" and self.config.filters.amplitude:
                current_tm, current_indices = self._apply_amplitude_filter(current_tm, current_indices)
            elif filter_type == "electrode" and self.config.filters.electrode:
                current_tm, current_indices = self._apply_electrode_filter(current_tm, current_indices)
            elif filter_type == "custom" and self.config.filters.custom:
                current_tm, current_indices = self._apply_custom_filter(current_tm, current_indices)
        
        self.filtered_templates = current_tm.templates
        self.selected_indices = current_indices
        
        self.logger.info(f"フィルタリング完了: {len(self.selected_indices)}/{self.template_maker.templates.shape[0]} クラスタ")
    
    def _apply_distance_filter(self, tm: TemplateMaker, indices: List[int]) -> tuple:
        """距離フィルタリングを適用"""
        distance_config = self.config.filters.distance
        
        if distance_config.max_distance is not None or distance_config.min_distance is not None:
            # 距離行列を計算
            distances = tm.calculate_template_distances(distance_config.method)
            
            # フィルタリング条件をチェック
            filtered_indices = []
            for i, cluster_idx in enumerate(indices):
                if distance_config.reference_cluster is not None:
                    # 特定のクラスタとの距離をチェック
                    distance = distances[cluster_idx, distance_config.reference_cluster]
                else:
                    # 全クラスタとの平均距離をチェック
                    cluster_distances = distances[cluster_idx, indices]
                    distance = np.mean(cluster_distances)
                
                # 距離条件をチェック
                if distance_config.min_distance is not None and distance < distance_config.min_distance:
                    continue
                if distance_config.max_distance is not None and distance > distance_config.max_distance:
                    continue
                
                filtered_indices.append(cluster_idx)
            
            # フィルタリングされたテンプレートで新しいTemplateMakerを作成
            if filtered_indices:
                filtered_templates = tm.templates[filtered_indices]
                new_tm = tm.create_filtered_template_maker(filtered_templates, filtered_indices)
                self.logger.info(f"距離フィルタリング: {len(filtered_indices)}/{len(indices)} クラスタ")
                return new_tm, filtered_indices
            else:
                self.logger.warning("距離フィルタリング条件に合致するクラスタが見つかりませんでした")
                return tm, indices
        
        return tm, indices
    
    def _apply_amplitude_filter(self, tm: TemplateMaker, indices: List[int]) -> tuple:
        """振幅フィルタリングを適用"""
        amplitude_config = self.config.filters.amplitude
        
        if amplitude_config.min_amplitude is not None or amplitude_config.max_amplitude is not None:
            filtered_templates, filtered_indices = tm.filter_by_amplitude_range(
                amplitude_config.min_amplitude or 0.0,
                amplitude_config.max_amplitude or float('inf')
            )
            
            if len(filtered_indices) > 0:
                new_tm = tm.create_filtered_template_maker(filtered_templates, filtered_indices)
                self.logger.info(f"振幅フィルタリング: {len(filtered_indices)}/{len(indices)} クラスタ")
                return new_tm, filtered_indices
            else:
                self.logger.warning("振幅フィルタリング条件に合致するクラスタが見つかりませんでした")
                return tm, indices
        
        return tm, indices
    
    def _apply_electrode_filter(self, tm: TemplateMaker, indices: List[int]) -> tuple:
        """電極数フィルタリングを適用"""
        electrode_config = self.config.filters.electrode
        
        if electrode_config.min_electrodes is not None or electrode_config.max_electrodes is not None:
            filtered_templates, filtered_indices = tm.filter_by_electrode_count(
                electrode_config.min_electrodes or 0,
                electrode_config.max_electrodes
            )
            
            if len(filtered_indices) > 0:
                new_tm = tm.create_filtered_template_maker(filtered_templates, filtered_indices)
                self.logger.info(f"電極数フィルタリング: {len(filtered_indices)}/{len(indices)} クラスタ")
                return new_tm, filtered_indices
            else:
                self.logger.warning("電極数フィルタリング条件に合致するクラスタが見つかりませんでした")
                return tm, indices
        
        return tm, indices
    
    def _apply_custom_filter(self, tm: TemplateMaker, indices: List[int]) -> tuple:
        """カスタムフィルタリングを適用"""
        custom_config = self.config.filters.custom
        
        if custom_config.conditions:
            # カスタム条件を実装（例として簡単な条件を実装）
            def custom_condition(cluster_idx: int, tm: TemplateMaker, **kwargs) -> bool:
                # ここでカスタム条件を実装
                # 例: ピーク振幅が平均以上
                peak_amplitudes = tm.get_peak_amplitudes()
                cluster_peak = np.max(peak_amplitudes[cluster_idx])
                mean_peak = np.mean(peak_amplitudes)
                return cluster_peak > mean_peak
            
            filtered_templates, filtered_indices = tm.filter_templates_by_condition(custom_condition)
            
            if len(filtered_indices) > 0:
                new_tm = tm.create_filtered_template_maker(filtered_templates, filtered_indices)
                self.logger.info(f"カスタムフィルタリング: {len(filtered_indices)}/{len(indices)} クラスタ")
                return new_tm, filtered_indices
            else:
                self.logger.warning("カスタムフィルタリング条件に合致するクラスタが見つかりませんでした")
                return tm, indices
        
        return tm, indices
    
    def _run_analysis(self) -> None:
        """分析を実行"""
        self.logger.info("詳細分析を実行します")
        
        if self.filtered_templates is not None:
            # フィルタリングされたテンプレートで分析
            analysis = self.template_maker.analyze_templates()
        else:
            # 元のテンプレートで分析
            analysis = self.template_maker.analyze_templates()
        
        self.logger.info(f"分析完了: {analysis['n_clusters']} クラスタ, {analysis['n_electrodes']} 電極")
    
    def _run_plots(self) -> None:
        """プロットを実行"""
        self.logger.info("プロットを実行します")
        
        try:
            if self.filtered_templates is not None:
                # フィルタリングされたテンプレートでプロット
                self.template_maker.plot_all_clusters_primary_electrodes(
                    max_clusters=self.config.plot.max_clusters
                )
            else:
                # 元のテンプレートでプロット
                self.template_maker.plot_all_clusters_primary_electrodes(
                    max_clusters=self.config.plot.max_clusters
                )
            
            self.logger.info("プロット完了")
        except ImportError:
            self.logger.warning("matplotlibがインストールされていないため、プロットをスキップします")
    
    def _save_results(self) -> None:
        """結果を保存"""
        self.logger.info("結果を保存します")
        
        # フィルタリングされたテンプレートを保存
        if self.config.output.save_filtered and self.filtered_templates is not None:
            save_path = Path(self.config.output.save_filtered)
            np.save(save_path, self.filtered_templates)
            self.logger.info(f"フィルタリングされたテンプレートを保存しました: {save_path}")
        
        # 分析結果を保存
        if self.config.output.save_analysis:
            save_path = Path(self.config.output.save_analysis)
            self.template_maker.save_analysis(save_path)
            self.logger.info(f"分析結果を保存しました: {save_path}")
        
        # プロットを保存
        if self.config.output.save_plots and self.config.plot and self.config.plot.enabled:
            save_dir = Path(self.config.output.save_plots)
            save_dir.mkdir(parents=True, exist_ok=True)
            # プロット保存の実装（必要に応じて）
            self.logger.info(f"プロット保存ディレクトリ: {save_dir}")


def run_from_config(config_path: Path) -> None:
    """設定ファイルから実行する便利な関数"""
    runner = TemplateMakerRunner(config_path)
    runner.run() 
