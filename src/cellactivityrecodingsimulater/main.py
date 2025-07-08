import numpy as np
import matplotlib.pyplot as plt
import carsIO
import tools
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    """メイン関数"""
    try:
        project_root = tools.getProjectRoot()
        logging.info(f"プロジェクトルート: {project_root}")

        # 設定ファイルの読み込み
        settings = carsIO.load_settings(project_root / "tests" / "data" / "test_settings.json")
        logging.info(f"設定ファイル読み込み完了: {settings}")
        
        # セルデータの読み込み
        cells = carsIO.load_cells(project_root / "tests" / "data" / settings.pathCell.name)
        logging.info(f"セルデータ読み込み完了: {len(cells)} cells")
        # サイトデータの読み込み
        sites = carsIO.load_sites(project_root / "tests" / "data" / settings.pathSite.name)
        logging.info(f"サイトデータ読み込み完了: {len(sites)} sites")
        
        if settings.noiseType == "truth":
            for site in sites:
                site.signalNoise = tools.getRecordingNoiseFromTruth(settings)
            logging.info("真のノイズを適用")
        elif settings.noiseType == "gaussian":
            for site in sites:
                site.signalNoise = tools.simulateRecordingNoise(settings, settings.noiseType)
            logging.info("ガウシアンノイズを適用")
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
            spikeTemplates = carsIO.load_spikeTemplates(project_root / "tests" / "data" / settings.pathSpikeList.name)
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
                tools.calcScaledSpikeAmp(cell, site, settings)
                tools.addSpikeToSignal(cell, site)
            site.signalFiltered = tools.getFilteredSignal(site.signalRaw, settings.fs, 300, 3000)
        logging.info("信号生成完了")

        # データの保存
        pathSaveDir = project_root / "tests" / "data" / settings.pathSaveDir
        pathSaveDir.mkdir(parents=True, exist_ok=True)
        carsIO.save_data(pathSaveDir, cells, sites)
        logging.info(f"データ保存完了: {pathSaveDir}")

        # データの表示
        plt.plot(sites[0].signalRaw)
        plt.show()
        logging.info("プロット表示完了")
    except Exception as e:
        logging.error(f"エラー発生: {e}", exc_info=True)

if __name__ == "__main__":
    main()
