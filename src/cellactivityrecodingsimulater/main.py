import numpy as np
import matplotlib.pyplot as plt
import carsIO
import tools

def main():
    """メイン関数"""
    project_root = tools.getProjectRoot()

    # 設定ファイルの読み込み
    settings = carsIO.load_settings(project_root /"src"/ "testData" / "test_settings.json")
    
    # セルデータの読み込み
    cells = carsIO.load_cells(project_root / "src" / settings.pathCell)
    # サイトデータの読み込み
    sites = carsIO.load_sites(project_root / "src" / settings.pathSite)
    
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
    else:
        spikeTemplates = carsIO.load_spikeTemplates(project_root / "src" / settings.pathSpikeList)

    # 各セルの処理
    for i, cell in enumerate(cells):
        spikeTimes = tools.simulateSpikeTimes(cell, settings)
        cell.spikeTimeList = spikeTimes
        if settings.spikeType == "gabor":
            cell.spikeTemp = tools.simulateSpikeTemplate(settings)
        else: 
            if settings.isRandomSelect:
                cell.spikeTemp = spikeTemplates[np.random.randint(0, len(spikeTemplates))]
            else:
                cell.spikeTemp = spikeTemplates[i]

    for site in sites:
        site.signalRaw = site.signalNoise.copy()
        for cell in cells:
            tools.calcScaledSpikeAmp(cell, site, settings)
            tools.addSpikeToSignal(cell, site)

    # データの保存
    pathSaveDir = project_root / "src" / settings.pathSaveDir
    pathSaveDir.mkdir(parents=True, exist_ok=True)
    carsIO.save_data(pathSaveDir, cells, sites)

    plt.plot(sites[0].signalRaw)
    plt.show()

    plt.plot(sites[0].signalNoise)
    plt.show()


    
    

if __name__ == "__main__":
    main()
