import numpy as np
import matplotlib.pyplot as plt
import carsIO
import tools
import logging
import argparse
import glob
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_single_experiment(condition_dir: Path, settings_file: Path, show_plot: bool = True):
    """単一の実験を実行する"""
    try:
        logging.info(f"=== 実験開始: {condition_dir.name} ===")
        
        # 設定ファイルの読み込み
        settings = carsIO.load_settings(settings_file)
        logging.info(f"設定ファイル読み込み完了: {settings}")
        
        # 乱数シードのセット
        import random
        random.seed(settings.random_seed)
        np.random.seed(settings.random_seed)
        
        # セルデータの読み込み（条件フォルダ内から）
        cell_file = condition_dir / "cells.json"
        if not cell_file.exists():
            cell_file = condition_dir / "cell.json"  # フォールバック
        cells = carsIO.load_cells(cell_file)
        logging.info(f"セルデータ読み込み完了: {len(cells)} cells")
        
        # サイトデータの読み込み（条件フォルダ内から）
        site_file = condition_dir / "probe.json"
        if not site_file.exists():
            site_file = condition_dir / "site.json"  # フォールバック
        sites = carsIO.load_sites(site_file)
        logging.info(f"サイトデータ読み込み完了: {len(sites)} sites")

        # 保存ディレクトリの作成
        if settings.pathSaveDir is None:
            pathSaveDir = condition_dir
        else:
            pathSaveDir = settings.pathSaveDir / condition_dir.name
        pathSaveDir.mkdir(parents=True, exist_ok=True)
        
        # ノイズの適用
        if settings.noiseType == "truth":
            for site in sites:
                site.signalNoise = tools.getRecordingNoiseFromTruth(settings)
            logging.info("真のノイズを適用")
        elif settings.noiseType == "normal" or settings.noiseType == "gaussian":
            for site in sites:
                site.signalNoise = tools.simulateRecordingNoise(settings, settings.noiseType)
            logging.info("乱数ノイズを適用")
        elif settings.noiseType == "none":
            for site in sites:
                site.signalNoise = np.zeros(int(settings.duration * settings.fs))
            logging.info("ノイズなしを適用")
        else:
            raise ValueError(f"Invalid noise type: {settings.noiseType}")
        
        # スパイクテンプレートの読み込み
        if settings.spikeType == "gabor":
            spikeTemplates = [tools.simulateSpikeTemplate(settings) for _ in range(len(cells))]
            logging.info("ガボールスパイクテンプレートを生成")
        elif settings.spikeType == "templates":
            # テンプレートファイルのパスを条件フォルダ内に変更
            template_file = condition_dir / "templates.json"
            if settings.pathSpikeList:
                template_file = condition_dir / settings.pathSpikeList.name
            spikeTemplates = carsIO.loadSpikeTemplates(template_file)
            logging.info("テンプレートスパイクテンプレートを読み込み")
        else:
            raise ValueError(f"Invalid spike type: {settings.spikeType}")
        
        # 各セルの処理
        for i, cell in enumerate(cells):
            cell.spikeTimeList = tools.simulateSpikeTimes(settings)
            cell.spikeTemp = spikeTemplates[i]
            for t in cell.spikeTimeList:
                cell.spikeAmpList.append(tools.calcSpikeAmp(settings))
            logging.info(f"セル{i}のスパイク: {cell.spikeTemp[0:10]}")
        logging.info("各セルのスパイク時刻・テンプレートを設定")

        # 信号生成
        for site in sites:
            site.signalRaw = site.signalNoise.copy()
            for cell in cells:
                # デバッグ情報を追加
                original_amp = cell.spikeAmpList[0] if cell.spikeAmpList else 0
                scaledSpikeAmpList = tools.calcScaledSpikeAmp(cell, site, settings)
                scaled_amp = scaledSpikeAmpList[0] if scaledSpikeAmpList else 0
                distance = tools.calcDistance(cell, site)
                #logging.info(f"セル{cell.id}: 元振幅={original_amp:.2f}, 距離={distance:.2f}, スケール後振幅={scaled_amp:.2f}")
                
                tools.addSpikeToSignal(cell, site, scaledSpikeAmpList)
            site.signalFiltered = tools.getFilteredSignal(site.signalRaw, settings.fs, 300, 3000)
        logging.info("信号生成完了")

        # データの保存
        carsIO.save_data(pathSaveDir, cells, sites)
        logging.info(f"データ保存完了: {pathSaveDir}")

        # プロット表示（オプション）
        if show_plot:
            plt.figure(figsize=(10, 8))
            
            # 元の信号
            tstart = 0
            tend = 3000
            plt.subplot(3, 1, 1)
            plt.plot(sites[0].signalRaw[tstart:tend])
            plt.title(f'Raw Signal - {condition_dir.name}')
            plt.ylabel('Amplitude')
            
            # フィルタ済み信号
            plt.subplot(3, 1, 2)
            plt.plot(sites[0].signalFiltered[tstart:tend])
            plt.title(f'Filtered Signal - {condition_dir.name}')
            plt.ylabel('Amplitude')

            plt.subplot(3, 1, 3)
            plt.plot(sites[0].signalNoise[tstart:tend])
            plt.title(f'Noise Signal - {condition_dir.name}')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            plt.show()
            
            # 信号の統計情報をログに出力
            logging.info(f"Raw signal - min: {np.min(sites[0].signalRaw):.2f}, max: {np.max(sites[0].signalRaw):.2f}, std: {np.std(sites[0].signalRaw):.2f}")
            logging.info(f"Filtered signal - min: {np.min(sites[0].signalFiltered):.2f}, max: {np.max(sites[0].signalFiltered):.2f}, std: {np.std(sites[0].signalFiltered):.2f}")
            logging.info("プロット表示完了")
        
        logging.info(f"=== 実験完了: {condition_dir.name} ===")
        return True
        
    except Exception as e:
        logging.error(f"実験でエラー発生: {condition_dir.name} - {e}", exc_info=True)
        return False

def main(example_dir: str, condition_pattern: str = "*", show_plot: bool = True):
    """メイン関数 - 複数の条件フォルダを実行"""
    try:
        example_dir = Path(example_dir)
        
        if not example_dir.exists():
            logging.error(f"実験ディレクトリが見つかりません: {example_dir}")
            return
        
        # 条件フォルダの検索
        condition_dirs = [d for d in example_dir.iterdir() 
                         if d.is_dir() and d.name != "results" and d.match(condition_pattern)]
        
        if not condition_dirs:
            logging.error(f"条件フォルダが見つかりません: {example_dir}/{condition_pattern}")
            return
        
        logging.info(f"実行する条件フォルダ数: {len(condition_dirs)}")
        for i, condition_dir in enumerate(condition_dirs, 1):
            logging.info(f"{i}. {condition_dir.name}")
        
        # 各条件フォルダを順次実行
        success_count = 0
        for i, condition_dir in enumerate(condition_dirs, 1):
            logging.info(f"\n[{i}/{len(condition_dirs)}] {condition_dir.name} を実行中...")
            
            # 設定ファイルの検索
            settings_files = list(condition_dir.glob("*.json"))
            if not settings_files:
                logging.error(f"設定ファイルが見つかりません: {condition_dir}")
                continue
            
            # 設定ファイルを優先順位で選択
            settings_file = None
            for filename in ["settings.json", "config.json", "setting.json"]:
                potential_file = condition_dir / filename
                if potential_file.exists():
                    settings_file = potential_file
                    break
            
            # 見つからない場合は最初のJSONファイルを使用
            if settings_file is None:
                settings_file = settings_files[0]
                logging.warning(f"明示的な設定ファイルが見つからないため、最初のJSONファイルを使用: {settings_file}")
            
            logging.info(f"使用する設定ファイル: {settings_file}")
            
            # 最後のフォルダ以外はプロットを無効化（オプション）
            current_show_plot = show_plot and (i == len(condition_dirs))
            
            if run_single_experiment(condition_dir, settings_file, current_show_plot):
                success_count += 1
            else:
                logging.error(f"実験が失敗しました: {condition_dir.name}")
        
        logging.info(f"\n=== 実行完了 ===")
        logging.info(f"成功: {success_count}/{len(condition_dirs)}")
        
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
