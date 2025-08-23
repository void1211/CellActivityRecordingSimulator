import matplotlib.pyplot as plt


def plot_signals(sites, condition_name):
    # プロット用の時間範囲を設定
    tstart = 0
    tend = min(15000, len(sites[0].signalRaw))  # 最初の1000サンプルを表示
    ch = 14
    plt.figure(figsize=(12, 8))

    # 生信号
    plt.subplot(6, 1, 1)
    plt.plot(sites[ch].signalRaw[tstart:tend])
    plt.title(f'Raw Signal - {condition_name}')
    plt.ylabel('Amplitude')

    # フィルタ済み信号
    plt.subplot(6, 1, 2)
    plt.plot(sites[ch].signalBGNoise[tstart:tend])
    plt.title(f'BG Noise Signal - {condition_name}')
    plt.ylabel('Amplitude')

    plt.subplot(6, 1, 3)
    plt.plot(sites[ch].signalRaw[tstart:tend] - sites[ch].signalNoise[tstart:tend])
    plt.title(f'Raw-Noise - {condition_name}')
    plt.ylabel('Amplitude')

    plt.subplot(6, 1, 4)
    plt.plot(sites[ch].signalNoise[tstart:tend])
    plt.title(f'Noise Signal - {condition_name}')
    plt.ylabel('Amplitude')

    plt.subplot(6, 1, 5)
    plt.plot(sites[ch].signalDrift[tstart:tend])
    plt.title(f'Drift Signal - {condition_name}')
    plt.ylabel('Amplitude')

    plt.subplot(6, 1, 6)
    plt.plot(sites[ch].signalPowerNoise[tstart:tend])
    plt.title(f'Power Line Noise - {condition_name}')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()