import numpy as np
import matplotlib.pyplot as plt

import logging
import argparse
from pathlib import Path
import os

from . import carsIO
from .simulate import simulateBackgroundActivity, simulateSpikeTimes, simulateSpikeTemplate, simulateNormalRandomNoise, simulateGaussianRandomNoise, simulateDrift, simulatePowerLineNoise
from .generate import generateNoiseCells, generate_similar_templates
from .calculate import calcSpikeAmp, calcScaledSpikeAmp, calcDistance
from .tools import addSpikeToSignal, filterSignal, makeSaveDir
from .plot.main import plot_main
# ベースディレクトリを固定
BASE_DIR = Path("simulations")
TEST_DIR = Path("test")

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_single_experiment(example_dir: Path, condition_name: str, show_plot: bool = True, test: bool = False):
    """単一の実験を実行する"""
    try:
        logging.info(f"=== 実験開始: {condition_name} ===")
        
        # 新しいディレクトリ構成に合わせてファイルパスを設定
        settings_file = example_dir / condition_name / "settings.json"
        cell_file = example_dir / condition_name / "cells.json"
        site_file = example_dir / condition_name / "probe.json"
        
        # 設定ファイルの読み込み
        if not settings_file.exists():
            logging.error(f"設定ファイルが見つかりません: {settings_file}")
            return False
        
        settings = carsIO.load_settings(settings_file)
        logging.info(f"設定ファイル読み込み完了: {settings.name}")
        
        # 設定の検証を実行
        logging.info("=== 設定検証開始 ===")
        validation_summary = settings.get_validation_summary()
        logging.info(validation_summary)
        
        # 重大なエラーがある場合は処理を停止
        errors = settings.validate_settings()
        if errors:
            logging.error("設定に重大なエラーがあります。処理を停止します。")
            return False
        
        logging.info("=== 設定検証完了 ===")
        
        # 乱数シードのセット
        import random
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed)
        
        # セルデータの読み込み
        if not cell_file.exists():
            logging.error(f"セルファイルが見つかりません: {cell_file}")
            return False
        cells = carsIO.load_cells(cell_file)
        logging.info(f"セルデータ読み込み完了: {len(cells)} cells")
        
        # サイトデータの読み込み
        if not site_file.exists():
            logging.error(f"サイトファイルが見つかりません: {site_file}")
            return False
        sites = carsIO.load_sites(site_file)
        logging.info(f"サイトデータ読み込み完了: {len(sites)} sites")

        # 保存ディレクトリの作成
        if settings.pathSaveDir is None:
            # ベースディレクトリを使用
            if test:
                pathSaveDir = example_dir / condition_name
            else:
                pathSaveDir = example_dir / condition_name
        
        saveDir = makeSaveDir(pathSaveDir)
        
        # ノイズの適用
        if settings.noiseType == "truth":
            for site in sites:
                site.signalBGNoise = carsIO.loadNoiseFile(settings.pathNoiseFile)
        elif settings.noiseType == "normal":    
            for site in sites:
                site.signalBGNoise = simulateNormalRandomNoise(settings.duration, settings.fs, settings.noiseAmp)
        elif settings.noiseType == "gaussian":
            for site in sites:
                site.signalBGNoise = simulateGaussianRandomNoise(settings.duration, settings.fs, settings.noiseAmp, settings.noiseLoc, settings.noiseScale)
        elif settings.noiseType == "model":
            # ノイズ細胞を生成してサイトに追加
            noise_cells = generateNoiseCells(settings.duration, settings.fs, sites, settings.margin, settings.density, settings.inviolableArea,
            spikeAmpMax=settings.spikeAmpMax, spikeAmpMin=settings.spikeAmpMin, spikeType=settings.spikeType,
            randType=settings.randType, gaborSigma=settings.gaborSigma, gaborf0=settings.gaborf0, gabortheta=settings.gabortheta,
            ms_before=settings.ms_before, ms_after=settings.ms_after, negative_amplitude=settings.negative_amplitude,
            positive_amplitude=settings.positive_amplitude, depolarization_ms=settings.depolarization_ms,
            repolarization_ms=settings.repolarization_ms, recovery_ms=settings.recovery_ms, smooth_ms=settings.smooth_ms,
            spikeWidth=settings.spikeWidth, rate=settings.avgSpikeRate, isRefractory=settings.isRefractory, refractoryPeriod=settings.refractoryPeriod)
            for site in sites:
                site.signalBGNoise = simulateBackgroundActivity(settings.duration, settings.fs, noise_cells, site, settings.attenTime)
        elif settings.noiseType == "none":
            for site in sites:
                site.signalBGNoise = np.zeros(int(settings.duration * settings.fs))
        else:
            raise ValueError(f"Invalid noise type: {settings.noiseType}")
        
        # スパイクテンプレートの読み込み
        if settings.spikeType == "gabor" or settings.spikeType == "exponential":
            if settings.enable_template_similarity_control:
                group_ids = list(set([cell.group for cell in cells]))
                for group_id in group_ids:
                    group_cells = [cell for cell in cells if cell.group == group_id]
                    spikeTemplates = generate_similar_templates(
                        settings.fs, len(group_cells), 
                        settings.spikeType, settings.randType, 
                        settings.gaborSigma, settings.gaborf0, settings.gabortheta, 
                        settings.ms_before, settings.ms_after, 
                        settings.negative_amplitude, settings.positive_amplitude, 
                        settings.depolarization_ms, settings.repolarization_ms, 
                        settings.recovery_ms, settings.smooth_ms, 
                        settings.spikeWidth,
                        settings.min_cosine_similarity, settings.max_cosine_similarity, 
                        settings.similarity_control_attempts)
                    for i, cell in enumerate(group_cells):
                        cell.spikeTemp = spikeTemplates[i]
                        cell.spikeTimeList = simulateSpikeTimes(settings.duration, settings.fs, settings.avgSpikeRate, settings.isRefractory, settings.refractoryPeriod)
                        logging.debug(f"cell{cell.id}.spikeTimeList: {len(cell.spikeTimeList)}")
                        for t in cell.spikeTimeList:
                            cell.spikeAmpList.append(calcSpikeAmp(settings.spikeAmpMax, settings.spikeAmpMin))
            else:
                spikeTemplates = [simulateSpikeTemplate(
                    settings.fs, settings.spikeType, settings.randType, settings.spikeWidth,
                    gaborSigma=settings.gaborSigma, gaborf0=settings.gaborf0, gabortheta=settings.gabortheta,
                    ms_before=settings.ms_before, ms_after=settings.ms_after,
                    negative_amplitude=settings.negative_amplitude, positive_amplitude=settings.positive_amplitude,
                    depolarization_ms=settings.depolarization_ms, 
                    repolarization_ms=settings.repolarization_ms, 
                    recovery_ms=settings.recovery_ms, 
                    smooth_ms=settings.smooth_ms
                ) for _ in range(len(cells))]
                for i, cell in enumerate(cells):
                    cell.spikeTemp = spikeTemplates[i]
                    cell.spikeTimeList = simulateSpikeTimes(settings.duration, settings.fs, settings.avgSpikeRate, settings.isRefractory, settings.refractoryPeriod)
                    for t in cell.spikeTimeList:
                        cell.spikeAmpList.append(calcSpikeAmp(settings.spikeAmpMax, settings.spikeAmpMin))

        elif settings.spikeType == "template":
            template_file = example_dir / settings.pathSpikeList.name
            if not template_file.exists():
                logging.error(f"テンプレートファイルが見つかりません: {template_file}")
                return False
            spikeTemplates = carsIO.load_spikeTemplates(template_file)
            for i, cell in enumerate(cells):
                cell.spikeTemp = spikeTemplates[i]
                cell.spikeTimeList = simulateSpikeTimes(settings.duration, settings.fs, settings.avgSpikeRate, settings.isRefractory, settings.refractoryPeriod)
                for t in cell.spikeTimeList:
                    cell.spikeAmpList.append(calcSpikeAmp(settings.spikeAmpMax, settings.spikeAmpMin))
        else:
            raise ValueError(f"Invalid spike type: {settings.spikeType}")

        # 信号生成
        logging.info("信号生成開始...")
        if settings.enable_drift:
            drift = simulateDrift(settings.duration, settings.fs, settings.drift_type)
        else:
            drift = np.zeros(int(settings.duration * settings.fs))
        if settings.enable_power_noise:
            powerLineNoise = simulatePowerLineNoise(settings.duration, settings.fs, settings.power_line_frequency, settings.power_noise_amplitude)
        else:
            powerLineNoise = np.zeros(int(settings.duration * settings.fs))

        for site in sites:        
            # メインの細胞のスパイクを追加
            spikeWithBGNoise = site.signalBGNoise.copy()
            for cell in cells:
                scaledSpikeAmpList = calcScaledSpikeAmp(cell.spikeAmpList, calcDistance(cell, site), settings.attenTime)
                spikeWithBGNoise = addSpikeToSignal(spikeWithBGNoise, cell.spikeTimeList, cell.spikeTemp, scaledSpikeAmpList)
            
            site.signalRaw = np.array(spikeWithBGNoise) + np.array(drift) + np.array(powerLineNoise)
            site.signalNoise = np.array(site.signalBGNoise) + np.array(powerLineNoise) + np.array(drift)
            site.signalDrift = np.array(drift)
            site.signalPowerNoise = np.array(powerLineNoise)
            site.signalFiltered = filterSignal(site.signalRaw, settings.fs, 300, 3000)

        # データの保存
        logging.info(f"保存先ディレクトリ: {saveDir}")
        logging.info(f"saveDirの型: {type(saveDir)}")
        logging.info(f"saveDirの絶対パス: {saveDir.absolute() if hasattr(saveDir, 'absolute') else 'N/A'}")
        
        if saveDir is None:
            logging.error("保存先ディレクトリがNoneです。データ保存をスキップします。")
            return False
            
        carsIO.save_data(saveDir, cells, sites, noise_cells=(noise_cells if settings.noiseType == "model" else None))
        logging.info(f"データ保存完了: {saveDir}")

        # プロット表示（オプション）
        if show_plot:
            plot_main(cells, noise_cells, sites, condition_name, show_plot=show_plot)

        logging.info(f"=== 実験完了: {condition_name} ===")
        return True
        
    except Exception as e:
        logging.error(f"実験でエラー発生: {condition_name} - {e}", exc_info=True)
        return False

def main(project_root: Path, example_dir: str, condition_pattern: str = "*", show_plot: bool = True, test: bool = False, debug: bool = False):
    """メイン関数 - 複数の条件を実行"""
    try:
        if debug:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        # ベースディレクトリからの相対パスとして扱う
        if test:
            example_dir = project_root / TEST_DIR / example_dir
        else:
            example_dir = project_root / BASE_DIR / example_dir
        
        if not example_dir.exists():
            logging.error(f"実験ディレクトリが見つかりません: {example_dir}")
            return

        
        # 条件ファイルの検索
        condition_files = list(example_dir.glob(f"{condition_pattern}"))
        
        if not condition_files:
            logging.error(f"条件ファイルが見つかりません: {example_dir}/{condition_pattern}")
            return
        
        # 条件名を抽出（.jsonを除く）
        condition_names = [f.stem for f in condition_files]
        
        logging.info(f"実行する条件数: {len(condition_names)}")
        for i, condition_name in enumerate(condition_names, 1):
            logging.info(f"{i}. {condition_name}")
        
        # 各条件を順次実行
        success_count = 0
        for i, condition_name in enumerate(condition_names, 1):
            logging.info(f"\n[{i}/{len(condition_names)}] {condition_name} を実行中...")
            
            # 最後の条件以外はプロットを無効化（オプション）
            current_show_plot = show_plot and (i == len(condition_names))
            
            if run_single_experiment(example_dir, condition_name, current_show_plot, test):
                success_count += 1
            else:
                logging.error(f"実験が失敗しました: {condition_name}")
        
        logging.info(f"\n=== 実行完了 ===")
        logging.info(f"成功: {success_count}/{len(condition_names)}")
        
    except Exception as e:
        logging.error(f"メイン処理でエラー発生: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Activity Recording Simulator")
    parser.add_argument("example_dir", help="実験ディレクトリのパス")
    parser.add_argument("--conditions", "-c", default="*", 
                       help="条件フォルダのパターン (デフォルト: *)")
    parser.add_argument("--no-plot", action="store_true", 
                       help="プロット表示を無効化")
    
    args = parser.parse_args()
    
    main(args.example_dir, args.conditions, not args.no_plot)
