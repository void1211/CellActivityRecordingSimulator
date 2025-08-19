import numpy as np
import matplotlib.pyplot as plt

import logging
import argparse
from pathlib import Path
import os

from . import carsIO
from .simulate import simulateBackgroundActivity, simulateSpikeTimes, simulateSpikeTemplate, simulateRandomNoise, simulateDrift, simulatePowerLineNoise
from .generate import generateNoiseCells, generate_similar_templates
from .calculate import calcSpikeAmp, calcScaledSpikeAmp, calcDistance
from .tools import addSpikeToSignal, filterSignal, makeSaveDir

# ベースディレクトリを固定
BASE_DIR = Path("simulations")
TEST_DIR = Path("test")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

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
                pathSaveDir = TEST_DIR / example_dir.name / condition_name
            else:
                pathSaveDir = BASE_DIR / example_dir.name / condition_name
        else:
            # 設定で指定された場合はベースディレクトリからの相対パス
            if test:
                pathSaveDir = TEST_DIR / settings.pathSaveDir / condition_name
            else:
                pathSaveDir = BASE_DIR / settings.pathSaveDir / condition_name
        
        saveDir = makeSaveDir(pathSaveDir)
        
        # ノイズの適用
        if settings.noiseType == "truth":
            for site in sites:
                site.signalBGNoise = carsIO.loadNoiseFile(settings.pathNoiseFile)
        elif settings.noiseType == "normal" or settings.noiseType == "gaussian":
            for site in sites:
                site.signalBGNoise = simulateRandomNoise(settings.duration, settings.fs, settings.noiseType, settings.noiseAmp)
        elif settings.noiseType == "model":
            # ノイズ細胞を生成してサイトに追加
            noise_cells = generateNoiseCells(settings.duration, settings.fs, sites, settings.margin, settings.density,
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
        drift = simulateDrift(settings.duration, settings.fs, settings.drift_type)
        powerLineNoise = simulatePowerLineNoise(settings.duration, settings.fs, settings.power_line_frequency, settings.power_noise_amplitude)

        for site in sites:        
            # メインの細胞のスパイクを追加
            spikeWithBGNoise = site.signalBGNoise
            for cell in cells:
                scaledSpikeAmpList = calcScaledSpikeAmp(cell.spikeAmpList, calcDistance(cell, site), settings.attenTime)
                spikeWithBGNoise = addSpikeToSignal(spikeWithBGNoise, cell.spikeTimeList, cell.spikeTemp, scaledSpikeAmpList)
            
            site.signalRaw = spikeWithBGNoise + drift + powerLineNoise
            site.signalNoise = site.signalBGNoise + powerLineNoise + drift
            site.signalDrift = drift
            site.signalPowerNoise = powerLineNoise
            site.signalFiltered = filterSignal(site.signalRaw, settings.fs, 300, 3000)

        # データの保存
        logging.info(f"保存先ディレクトリ: {saveDir}")
        logging.info(f"saveDirの型: {type(saveDir)}")
        logging.info(f"saveDirの絶対パス: {saveDir.absolute() if hasattr(saveDir, 'absolute') else 'N/A'}")
        
        if saveDir is None:
            logging.error("保存先ディレクトリがNoneです。データ保存をスキップします。")
            return False
            
        carsIO.save_data(saveDir, cells, sites, noise_cells)
        logging.info(f"データ保存完了: {saveDir}")

        # プロット表示（オプション）
        if show_plot:
            
            # プロット用の時間範囲を設定
            tstart = 0
            tend = min(300000, len(sites[0].signalRaw))  # 最初の1000サンプルを表示
            
            plt.figure(figsize=(12, 8))
            
            # 生信号
            plt.subplot(6, 1, 1)
            plt.plot(sites[0].signalRaw[tstart:tend])
            plt.title(f'Raw Signal - {condition_name}')
            plt.ylabel('Amplitude')
            
            # フィルタ済み信号
            plt.subplot(6, 1, 2)
            plt.plot(sites[0].signalFiltered[tstart:tend])
            plt.title(f'Filtered Signal - {condition_name}')
            plt.ylabel('Amplitude')

            plt.subplot(6, 1, 3)
            plt.plot(sites[0].signalRaw[tstart:tend] - sites[0].signalFiltered[tstart:tend])
            plt.title(f'BG Noise Signal - {condition_name}')
            plt.ylabel('Amplitude')

            plt.subplot(6, 1, 4)
            plt.plot(sites[0].signalNoise[tstart:tend])
            plt.title(f'Noise Signal - {condition_name}')
            plt.ylabel('Amplitude')

            plt.subplot(6, 1, 5)
            plt.plot(sites[0].signalDrift[tstart:tend])
            plt.title(f'Drift Signal - {condition_name}')
            plt.ylabel('Amplitude')

            plt.subplot(6, 1, 6)
            plt.plot(sites[0].signalPowerNoise[tstart:tend])
            plt.title(f'Power Line Noise - {condition_name}')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            plt.show()
            
            # ISIプロットを表示
            # if len(cells) > 0:
            #     tools.plotMultipleCellISI(cells, settings.fs)
                
            #     # 最初の細胞の詳細ISIプロットも表示
            #     if len(cells) > 0 and len(cells[0].spikeTimeList) > 1:
            #         tools.plotISI(cells[0].spikeTimeList, settings.fs, cell_id=cells[0].id)
                    
            #         # 不応期効果の可視化
            #         if settings.isRefractory:
            #             tools.plotRefractoryEffect(cells[0].spikeTimeList, settings.fs, 
            #                                     settings.refractoryPeriod, cell_id=cells[0].id)
        
        logging.info(f"=== 実験完了: {condition_name} ===")
        return True
        
    except Exception as e:
        logging.error(f"実験でエラー発生: {condition_name} - {e}", exc_info=True)
        return False

def main(project_root: Path, example_dir: str, condition_pattern: str = "*", show_plot: bool = True, test: bool = False):
    """メイン関数 - 複数の条件を実行"""
    try:
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
