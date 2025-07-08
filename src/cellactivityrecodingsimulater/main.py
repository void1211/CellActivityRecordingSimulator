import numpy as np
import matplotlib.pyplot as plt
import carsIO
import tools

def main():
    """メイン関数"""
    project_root = tools.getProjectRoot()

    # 設定ファイルの読み込み
    settings = carsIO.load_settings(project_root / "tests" / "data" / "test_settings.json")
    
    # セルデータの読み込み
    cells = carsIO.load_cells(project_root / "tests" / "data" / settings.pathCell.name)
    # サイトデータの読み込み
    sites = carsIO.load_sites(project_root / "tests" / "data" / settings.pathSite.name)
    
    if settings.noiseType == "truth":
        for site in sites:
            site.signalNoise = tools.getRecordingNoiseFromTruth(settings)
    elif settings.noiseType == "gaussian":
        for site in sites:
            site.signalNoise = tools.simulateRecordingNoise(settings, settings.noiseType)
    elif settings.noiseType == "none":
        for site in sites:
            site.signalNoise = np.zeros(int(settings.duration * settings.fs))
    else:
        raise ValueError(f"Invalid noise type: {settings.noiseType}")
    
    # スパイクテンプレートの読み込み
    if settings.spikeType == "gabor":
        spikeTemplates = [tools.simulateSpikeTemplate(settings) for _ in range(len(cells))]
    elif settings.spikeType == "templates":
        spikeTemplates = carsIO.load_spikeTemplates(project_root / "src" / settings.pathSpikeList)
    else:
        raise ValueError(f"Invalid spike type: {settings.spikeType}")

    # 各セルの処理
    for i, cell in enumerate(cells):
        cell.spikeTimeList = tools.simulateSpikeTimes(cell, settings)
        cell.spikeTemp = spikeTemplates[i]

    for site in sites:
        site.signalRaw = site.signalNoise.copy()
        for cell in cells:
            tools.calcScaledSpikeAmp(cell, site, settings)
            tools.addSpikeToSignal(cell, site)

    # データの保存
    pathSaveDir = project_root / "tests" / "data" / settings.pathSaveDir
    pathSaveDir.mkdir(parents=True, exist_ok=True)
    carsIO.save_data(pathSaveDir, cells, sites)

    # データの表示
    plt.plot(sites[0].signalRaw)
    plt.show()

if __name__ == "__main__":
    main()
