import numpy as np
import matplotlib.pyplot as plt
import carsIO
import tools
import logging
import argparse
import glob
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_single_experiment(example_dir: Path, settings_file: Path, show_plot: bool = True):
    """単一の実験を実行する"""
    try:
        logging.info(f"=== 実験開始: {settings_file.name} ===")
        
        # 設定ファイルの読み込み
        settings = carsIO.load_settings(settings_file)
        logging.info(f"設定ファイル読み込み完了: {settings}")
        
        # セルデータの読み込み
        cells = carsIO.load_cells(example_dir / "cell" / settings.pathCell.name)
        logging.info(f"セルデータ読み込み完了: {len(cells)} cells")
        
        # サイトデータの読み込み
        sites = carsIO.load_sites(example_dir / "probe" / settings.pathSite.name)
        logging.info(f"サイトデータ読み込み完了: {len(sites)} sites")

        # 保存ディレクトリの作成
        if settings.pathSaveDir is None:
            pathSaveDir = example_dir / "results" / settings.name
        else:
            pathSaveDir = settings.pathSaveDir / settings.name
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
            spikeTemplates = carsIO.loadSpikeTemplates(example_dir / "template" / settings.pathSpikeList.name)
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
            plt.title(f'Raw Signal - {settings.name}')
            plt.ylabel('Amplitude')
            
            # フィルタ済み信号
            plt.subplot(3, 1, 2)
            plt.plot(sites[0].signalFiltered[tstart:tend])
            plt.title(f'Filtered Signal - {settings.name}')
            plt.ylabel('Amplitude')

            plt.subplot(3, 1, 3)
            plt.plot(sites[0].signalNoise[tstart:tend])
            plt.title(f'Noise Signal - {settings.name}')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            plt.show()
            
            # 信号の統計情報をログに出力
            logging.info(f"Raw signal - min: {np.min(sites[0].signalRaw):.2f}, max: {np.max(sites[0].signalRaw):.2f}, std: {np.std(sites[0].signalRaw):.2f}")
            logging.info(f"Filtered signal - min: {np.min(sites[0].signalFiltered):.2f}, max: {np.max(sites[0].signalFiltered):.2f}, std: {np.std(sites[0].signalFiltered):.2f}")
            logging.info("プロット表示完了")
        
        logging.info(f"=== 実験完了: {settings_file.name} ===")
        return True
        
    except Exception as e:
        logging.error(f"実験でエラー発生: {settings_file.name} - {e}", exc_info=True)
        return False

def main(example_dir: str, settings_pattern: str = "*.json", show_plot: bool = True):
    """メイン関数 - 複数の設定ファイルを実行"""
    try:
        example_dir = Path(example_dir)
        settings_dir = example_dir / "settings"
        
        if not settings_dir.exists():
            logging.error(f"設定ディレクトリが見つかりません: {settings_dir}")
            return
        
        # 設定ファイルの検索
        settings_files = list(settings_dir.glob(settings_pattern))
        
        if not settings_files:
            logging.error(f"設定ファイルが見つかりません: {settings_dir}/{settings_pattern}")
            return
        
        logging.info(f"実行する設定ファイル数: {len(settings_files)}")
        for i, settings_file in enumerate(settings_files, 1):
            logging.info(f"{i}. {settings_file.name}")
        
        # 各設定ファイルを順次実行
        success_count = 0
        for i, settings_file in enumerate(settings_files, 1):
            logging.info(f"\n[{i}/{len(settings_files)}] {settings_file.name} を実行中...")
            
            # 最後のファイル以外はプロットを無効化（オプション）
            current_show_plot = show_plot and (i == len(settings_files))
            
            if run_single_experiment(example_dir, settings_file, current_show_plot):
                success_count += 1
            else:
                logging.error(f"実験が失敗しました: {settings_file.name}")
        
        logging.info(f"\n=== 実行完了 ===")
        logging.info(f"成功: {success_count}/{len(settings_files)}")
        
    except Exception as e:
        logging.error(f"メイン処理でエラー発生: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cell Activity Recording Simulator")
    parser.add_argument("example_dir", help="実験ディレクトリのパス")
    parser.add_argument("--settings", "-s", default="*.json", 
                       help="設定ファイルのパターン (デフォルト: *.json)")
    parser.add_argument("--no-plot", action="store_true", 
                       help="プロット表示を無効化")
    
    args = parser.parse_args()
    
    main(args.example_dir, args.settings, not args.no_plot)
