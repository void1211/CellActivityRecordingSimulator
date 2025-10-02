import matplotlib.pyplot as plt
import numpy as np  
from .spike_waveforms import plot_main as plot_spike_waveforms_main

def plot_signals(sites, condition_name, start=0, end=15000, ch=0, dynamic=False):
    # プロット用の時間範囲を設定
    tstart = start
    tend = min(end, len(sites[0].get_signal("raw")))  # 最初の1000サンプルを表示
    ch = ch
    fig = plt.figure(figsize=(12, 8))

    # 生信号
    ax_raw = fig.add_subplot(6, 1, 1)
    line_raw, = ax_raw.plot(sites[ch].get_signal("raw")[tstart:tend])
    ax_raw.set_title(f'Raw Signal - {condition_name}')
    ax_raw.set_ylabel('Amplitude')

    # フィルタ済み信号
    ax_bgnoise = fig.add_subplot(6, 1, 2)
    line_bgnoise, = ax_bgnoise.plot(sites[ch].get_signal("background")[tstart:tend])
    ax_bgnoise.set_title(f'BG Noise Signal - {condition_name}')
    ax_bgnoise.set_ylabel('Amplitude')

    ax_rawnoise = fig.add_subplot(6, 1, 3)
    line_rawnoise, = ax_rawnoise.plot(sites[ch].get_signal("raw")[tstart:tend] - sites[ch].get_signal("noise")[tstart:tend])
    ax_rawnoise.set_title(f'Raw-Noise - {condition_name}')
    ax_rawnoise.set_ylabel('Amplitude')

    ax_noise = fig.add_subplot(6, 1, 4)
    line_noise, = ax_noise.plot(sites[ch].get_signal("noise")[tstart:tend])
    ax_noise.set_title(f'Noise Signal - {condition_name}')
    ax_noise.set_ylabel('Amplitude')

    ax_drift = fig.add_subplot(6, 1, 5)
    line_drift, = ax_drift.plot(sites[ch].get_signal("drift")[tstart:tend])
    ax_drift.set_title(f'Drift Signal - {condition_name}')
    ax_drift.set_ylabel('Amplitude')

    ax_powernoise = fig.add_subplot(6, 1, 6)
    line_powernoise, = ax_powernoise.plot(sites[ch].get_signal("power")[tstart:tend])
    ax_powernoise.set_title(f'Power Line Noise - {condition_name}')
    ax_powernoise.set_ylabel('Amplitude')

    if dynamic:
        while True:
            tstart += 1000
            # if tstart >= len(sites[ch].signalRaw):
            #     tstart = 0
            tend = tstart + 15000
            # if tend >= len(sites[ch].signalRaw):
            #     tend = end
            # Check for 'q' key press without blocking
            if plt.waitforbuttonpress(timeout=0.001):
                key = plt.gcf().canvas.key_press_event
                if key is not None and key.key == 'q':
                    plt.close()
                    break
            line_raw.set_data(np.arange(tstart, tend), sites[ch].get_signal("raw")[tstart:tend])
            line_bgnoise.set_data(np.arange(tstart, tend), sites[ch].get_signal("background")[tstart:tend])
            line_rawnoise.set_data(np.arange(tstart, tend), sites[ch].get_signal("raw")[tstart:tend] - sites[ch].get_signal("noise")[tstart:tend])
            line_noise.set_data(np.arange(tstart, tend), sites[ch].get_signal("noise")[tstart:tend])
            line_drift.set_data(np.arange(tstart, tend), sites[ch].get_signal("drift")[tstart:tend])
            line_powernoise.set_data(np.arange(tstart, tend), sites[ch].get_signal("power")[tstart:tend])

            ax_raw.set_xlim(tstart, tend)
            ax_bgnoise.set_xlim(tstart, tend)
            ax_rawnoise.set_xlim(tstart, tend)
            ax_noise.set_xlim(tstart, tend)
            ax_drift.set_xlim(tstart, tend)
            ax_powernoise.set_xlim(tstart, tend)
            
            ax_raw.set_ylim(np.min(sites[ch].get_signal("raw")), np.max(sites[ch].get_signal("raw")))
            ax_bgnoise.set_ylim(np.min(sites[ch].get_signal("background")), np.max(sites[ch].get_signal("background")))
            ax_rawnoise.set_ylim(np.min(sites[ch].get_signal("raw")[tstart:tend] - sites[ch].get_signal("noise")[tstart:tend]), np.max(sites[ch].get_signal("raw")[tstart:tend] - sites[ch].get_signal("noise")[tstart:tend]))
            ax_noise.set_ylim(np.min(sites[ch].get_signal("noise")[tstart:tend]), np.max(sites[ch].get_signal("noise")[tstart:tend]))
            ax_drift.set_ylim(np.min(sites[ch].get_signal("drift")[tstart:tend]), np.max(sites[ch].get_signal("drift")[tstart:tend]))
            ax_powernoise.set_ylim(np.min(sites[ch].get_signal("power")[tstart:tend]), np.max(sites[ch].get_signal("power")[tstart:tend]))
            
            plt.pause(0.01)
            fig.tight_layout()
    else:
        fig.tight_layout()
    return fig

def plot_templates(cells, condition_name, ids='all'):
    if ids == 'all':
        cells = cells
    else:
        for id in ids:
            for cell in cells:
                if cell.id == id:
                    cells.append(cell)
                    break
    fig = plt.figure(figsize=(12, 8))
    for i, cell in enumerate(cells):
        # matplotlibのサブプロット番号は1から始まる必要があるため、i+1を使用
        ax = fig.add_subplot(len(cells) // 2, 2, i + 1)
        ax.plot(cell.spikeTemp)
        ax.set_title(f'Template - {condition_name} (Cell ID: {cell.id})')
    fig.tight_layout()
    return fig

def plot_cells(cells, noise_cells, sites,condition_name):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cell in cells:
        ax.scatter(cell.x, cell.y, cell.z, marker='o', color='blue')
    for cell in noise_cells:
        ax.scatter(cell.x, cell.y, cell.z, marker='x', color='red')
    for site in sites:
        ax.scatter(site.x, site.y, site.z, marker='*', color='green')
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(f'Cells - {condition_name}')
    return fig

def plot_main(cells, noise_cells, sites, condition_name, save_dir=None):
    """メインのプロット関数 - 既存のプロットと新しい波形プロットを統合"""
    
    figures = {}
    
    try:
        # 1. 既存の信号プロット
        print("信号プロットの作成中...")
        signal_fig = plot_signals(sites, condition_name)
        figures['signals'] = signal_fig
        
        # 2. 既存のテンプレートプロット
        print("スパイクテンプレートプロットの作成中...")
        template_fig = plot_templates(cells, condition_name)
        figures['templates'] = template_fig
        
        # 3. 新しいスパイク波形プロット
        print("スパイク波形プロットの作成中...")
        waveform_figures = plot_spike_waveforms_main(cells, sites, condition_name, save_dir)
        if waveform_figures:
            figures.update(waveform_figures)
        
        # 4. 細胞配置プロット
        print("細胞配置プロットの作成中...")
        cell_fig = plot_cells(cells, noise_cells, sites, condition_name)
        figures['cells'] = cell_fig

        plt.show()
        
        print("プロット作成完了")
        
    except Exception as e:
        print(f"プロット作成でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    return figures

