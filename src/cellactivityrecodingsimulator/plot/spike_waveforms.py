#!/usr/bin/env python3
"""
サイトごとの信号から細胞の発火時刻をもとに波形を切り出して重ね書きするプログラム

処理の軽量化のため、細胞がある場所の周囲のサイトのみを描画
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple
from ..calculate import calculate_distance_two_objects

class TemplateWaveformVisualizer:
    """テンプレートベースの波形可視化クラス（元のコードを参考）"""
    
    def __init__(self, fs: float = 30000, window_ms: float = 4.0, 
                 max_distance: float = 200.0, max_waveforms: int = 100):
        """
        Args:
            fs: サンプリングレート (Hz)
            window_ms: 波形切り出しウィンドウ (ms)
            max_distance: 細胞から描画するサイトの最大距離 (μm)
            max_waveforms: 1つのサイトに描画する最大波形数
        """
        self.fs = fs
        self.window_ms = window_ms
        self.max_distance = max_distance
        self.max_waveforms = max_waveforms
        
        # 波形切り出しの設定
        self.window_size = int(window_ms / 1000 * fs)
        self.half_window = self.window_size // 2
        
        logging.info(f"テンプレート波形可視化設定: ウィンドウ={window_ms}ms, サンプル数={self.window_size}")
    
    def extract_spike_waveforms(self, signal: np.ndarray, spike_times: List[int], 
                               contact_id: int, unit_id: int) -> List[np.ndarray]:
        """指定されたサイトの信号からスパイク波形を切り出す"""
        waveforms = []
        
        for i, spike_time in enumerate(spike_times):
            if i >= self.max_waveforms:  # 最大波形数で制限
                break
                
            # スパイク時刻を中心とした波形切り出し
            start_idx = spike_time - self.half_window
            end_idx = start_idx + self.window_size
            
            # 範囲チェック
            if start_idx >= 0 and end_idx <= len(signal):
                waveform = signal[start_idx:end_idx]
                if len(waveform) == self.window_size:
                    waveforms.append(waveform)
            else:
                logging.debug(f"サイト{contact_id}, 細胞{unit_id}: スパイク{i}が範囲外 (時刻={spike_time}, 範囲=[{start_idx}:{end_idx}])")
        
        return waveforms
    
    def find_nearby_contacts(self, unit: Any, contacts: List[Any]) -> List[Tuple[Any, float]]:
        """細胞の近くにあるサイトを見つける"""
        nearby_contacts = []
        
        for contact in contacts:
            distance = calculate_distance_two_objects(unit, contact)
            if distance <= self.max_distance:
                nearby_contacts.append((contact, distance))
        
        # 距離順にソート
        nearby_contacts.sort(key=lambda x: x[1])
        
        logging.debug(f"細胞{unit.id}: 近くのサイト数={len(nearby_contacts)}, 最大距離={self.max_distance}μm")
        for contact, dist in nearby_contacts[:5]:  # 最初の5個のみ表示
            logging.debug(f"  サイト{contact.id}: 距離={dist:.1f}μm")
        
        return nearby_contacts
    
    def find_best_channel(self, unit: Any, contacts: List[Any]) -> int:
        """細胞に最も近いサイト（ベストチャンネル）のインデックスを見つける"""
        nearby_contacts = self.find_nearby_contacts(unit, contacts)
        if not nearby_contacts:
            return 0
        
        # 最も近いサイトのインデックスを返す
        best_contact = nearby_contacts[0][0]
        return contacts.index(best_contact)
    
    def create_waveform_templates(self, units: List[Any], contacts: List[Any]) -> Tuple[np.ndarray, List[int]]:
        """各細胞の波形テンプレートとベストチャンネルを作成"""
        templates = []
        chan_best = []
        
        for unit in units:
            # 近くのサイトを見つける
            nearby_contacts = self.find_nearby_contacts(unit, contacts)
            if not nearby_contacts:
                # 近くのサイトがない場合は空のテンプレート
                empty_template = np.zeros((self.window_size, len(contacts)))
                templates.append(empty_template)
                chan_best.append(0)
                continue
            
            # 各サイトの波形を切り出してテンプレートを作成
            template = np.zeros((self.window_size, len(contacts)))
            
            for contact_idx, contact in enumerate(contacts):
                waveforms = self.extract_spike_waveforms(
                    contact.get_signal("raw"), unit.spikeTimeList, contact.id, unit.id
                )
                
                if waveforms:
                    # 平均波形を計算
                    if len(waveforms) > 1:
                        template[:, contact_idx] = np.mean(waveforms, axis=0)
                    else:
                        template[:, contact_idx] = waveforms[0]
            
            templates.append(template)
            
            # ベストチャンネル（最も近いサイト）を記録
            best_chan = self.find_best_channel(unit, contacts)
            chan_best.append(best_chan)
        
        return np.array(templates), chan_best
    
    def plot_probe_waveforms(self, units: List[Any], contacts: List[Any], 
                            condition_name: str = "", save_dir: Path = None, 
                            max_units_per_plot: int = 40, channels_per_unit: int = 16) -> plt.Figure:
        """プローブ位置ベースで波形を可視化（元のコードを参考）"""
        
        # プローブの位置情報を取得
        xc = np.array([contact.x for contact in contacts])
        yc = np.array([contact.y for contact in contacts])
        
        # 波形テンプレートとベストチャンネルを作成
        templates, chan_best = self.create_waveform_templates(units, contacts)
        
        # 全細胞の波形をプロット
        print(f'~~~~~~~~~~~~~~ All units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('title = number of spikes from each unit')
        
        if not units:
            print("No units found")
            return None
        
        fig = plt.figure(figsize=(12, 3), dpi=150)
        grid = plt.GridSpec(2, 20, figure=fig, hspace=0.25, wspace=0.5)
        
        # 各ユニットの波形をプロット
        for k in range(min(max_units_per_plot, len(units))):
            # ランダムにユニットを選択
            if len(units) > 1:
                wi = np.random.randint(len(units))
            else:
                wi = 0
            
            # テンプレートとベストチャンネルを取得
            wv = templates[wi].copy()
            cb = chan_best[wi]
            nsp = len(units[wi].spikeTimeList)
            
            # サブプロットを作成
            ax = fig.add_subplot(grid[k//20, k%20])
            
            # チャンネル数を制限（ベストチャンネルを中心に）
            n_chan = wv.shape[-1]
            ic0 = max(0, cb - channels_per_unit//2)
            ic1 = min(n_chan, cb + channels_per_unit//2)
            wv = wv[:, ic0:ic1]
            x0, y0 = xc[ic0:ic1], yc[ic0:ic1]
            
            # 各チャンネルの波形を描画
            amp = 4
            for ii, (xi, yi) in enumerate(zip(x0, y0)):
                t = np.arange(-wv.shape[0]//2, wv.shape[0]//2, 1, dtype='float32')
                t /= wv.shape[0] / 20
                ax.plot(xi + t, yi + wv[:, ii] * amp, lw=0.5, color='k')
            
            ax.set_title(f'{nsp}', fontsize='small')
            ax.axis('off')
        
        # 使用されていないサブプロットを非表示
        for k in range(len(units), max_units_per_plot):
            ax = fig.add_subplot(grid[k//20, k%20])
            ax.set_visible(False)
        
        plt.suptitle(f'All Unitss - {condition_name}', fontsize=14)
        
        # 保存
        if save_dir is not None:
            save_path = save_dir / f'probe_waveforms_{condition_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"プローブ波形プロットを保存しました: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_unit_waveforms_on_probe(self, units: List[Any], contacts: List[Any], 
                                    condition_name: str = "", save_dir: Path = None,
                                    max_units: int = 20) -> plt.Figure:
        """特定の細胞の波形をプローブ位置に配置して表示"""
        
        # プローブの位置情報を取得
        xc = np.array([contact.x for contact in contacts])
        yc = np.array([contact.y for contact in contacts])
        
        # 細胞数を制限
        n_units = min(max_units, len(units))
        
        # サブプロットのレイアウトを決定
        cols = min(5, n_units)
        rows = (n_units + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if n_units == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(rows, cols)
        
        fig.suptitle(f'Unit Waveforms on Probe - {condition_name}', fontsize=16)
        
        for unit_idx in range(n_units):
            if unit_idx >= rows * cols:
                break
            
            # 行と列のインデックスを計算
            row_idx = unit_idx // cols
            col_idx = unit_idx % cols
            
            # axesから正しいサブプロットを取得
            if rows == 1 and cols == 1:
                ax = axes[0]
            elif rows == 1:
                ax = axes[col_idx]
            elif cols == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            unit = units[unit_idx]
            
            # 近くのサイトを見つける
            nearby_contacts = self.find_nearby_contacts(unit, contacts)
            if not nearby_contacts:
                ax.text(0.5, 0.5, f'Unit {unit.id}\nNo nearby contacts', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Unit {unit.id}')
                continue
            
            # 各サイトの波形を描画
            amp = 3  # 振幅のスケール係数
            colors = plt.cm.viridis(np.linspace(0, 1, len(nearby_contacts)))
            
            for contact_idx, (contact, distance) in enumerate(nearby_contacts):
                # スパイク波形を切り出し
                waveforms = self.extract_spike_waveforms(
                    contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                )
                
                if not waveforms:
                    continue
                
                # 平均波形を計算
                if len(waveforms) > 1:
                    wv = np.mean(waveforms, axis=0)
                else:
                    wv = waveforms[0]
                
                # 時間軸を作成
                t = np.arange(-len(wv)//2, len(wv)//2, 1, dtype='float32')
                t /= len(wv) / 15  # スケーリング
                
                # プローブ位置に波形を配置
                xi, yi = contact.x, contact.y
                color = colors[contact_idx]
                ax.plot(xi + t, yi + wv * amp, lw=1, color=color, alpha=0.8,
                       label=f'Contact {contact.id} (d={distance:.0f}μm)')
            
            # 軸の設定
            ax.set_title(f'Unit {unit.id} - {len(nearby_contacts)} contacts')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='upper right')
            
            # プローブの範囲を表示
            ax.set_xlim(min(xc) - 50, max(xc) + 50)
            ax.set_ylim(min(yc) - 50, max(yc) + 50)
        
        # 使用されていないサブプロットを非表示
        for i in range(n_units, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            if rows == 1 and cols == 1:
                axes[0].set_visible(False)
            elif rows == 1:
                axes[col_idx].set_visible(False)
            elif cols == 1:
                axes[row_idx].set_visible(False)
            else:
                axes[row_idx, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存
        if save_dir is not None:
            save_path = save_dir / f'unit_waveforms_on_probe_{condition_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"細胞波形プローブプロットを保存しました: {save_path}")
        
        return fig

class ProbeWaveformVisualizer:
    """プローブ位置ベースの波形可視化クラス（旧版）"""
    
    def __init__(self, fs: float = 30000, window_ms: float = 4.0, 
                 max_distance: float = 200.0, max_waveforms: int = 100):
        """
        Args:
            fs: サンプリングレート (Hz)
            window_ms: 波形切り出しウィンドウ (ms)
            max_distance: 細胞から描画するサイトの最大距離 (μm)
            max_waveforms: 1つのサイトに描画する最大波形数
        """
        self.fs = fs
        self.window_ms = window_ms
        self.max_distance = max_distance
        self.max_waveforms = max_waveforms
        
        # 波形切り出しの設定
        self.window_size = int(window_ms / 1000 * fs)
        self.half_window = self.window_size // 2
        
        logging.info(f"プローブ波形可視化設定: ウィンドウ={window_ms}ms, サンプル数={self.window_size}")
    
    def extract_spike_waveforms(self, signal: np.ndarray, spike_times: List[int], 
                               contact_id: int, unit_id: int) -> List[np.ndarray]:
        """指定されたサイトの信号からスパイク波形を切り出す"""
        waveforms = []
        
        for i, spike_time in enumerate(spike_times):
            if i >= self.max_waveforms:  # 最大波形数で制限
                break
                
            # スパイク時刻を中心とした波形切り出し
            start_idx = spike_time - self.half_window
            end_idx = start_idx + self.window_size
            
            # 範囲チェック
            if start_idx >= 0 and end_idx <= len(signal):
                waveform = signal[start_idx:end_idx]
                if len(waveform) == self.window_size:
                    waveforms.append(waveform)
            else:
                logging.debug(f"サイト{contact_id}, 細胞{unit_id}: スパイク{i}が範囲外 (時刻={spike_time}, 範囲=[{start_idx}:{end_idx}])")
        
        return waveforms
    
    def find_nearby_contacts(self, unit: Any, contacts: List[Any]) -> List[Tuple[Any, float]]:
        """細胞の近くにあるサイトを見つける"""
        nearby_contacts = []
        
        for contact in contacts:
            distance = calculate_distance_two_objects(unit, contact)
            if distance <= self.max_distance:
                nearby_contacts.append((contact, distance))
        
        # 距離順にソート
        nearby_contacts.sort(key=lambda x: x[1])
        
        logging.debug(f"細胞{unit.id}: 近くのサイト数={len(nearby_contacts)}, 最大距離={self.max_distance}μm")
        for contact, dist in nearby_contacts[:5]:  # 最初の5個のみ表示
            logging.debug(f"  サイト{contact.id}: 距離={dist:.1f}μm")
        
        return nearby_contacts
    
    def plot_probe_waveforms(self, units: List[Any], contacts: List[Any], 
                            condition_name: str = "", save_dir: Path = None, 
                            max_units_per_plot: int = 40, channels_per_unit: int = 16) -> plt.Figure:
        """プローブ位置ベースで波形を可視化"""
        
        # プローブの位置情報を取得
        xc = np.array([contact.x for contact in contacts])
        yc = np.array([contact.y for contact in contacts])
        
        # 全細胞の波形をプロット
        print(f'~~~~~~~~~~~~~~ All units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('title = number of spikes from each unit')
        
        if not units:
            print("No units found")
            return None
        
        fig = plt.figure(figsize=(12, 3), dpi=150)
        grid = plt.GridSpec(2, 20, figure=fig, hspace=0.25, wspace=0.5)
        
        # 各ユニットの波形をプロット
        for k in range(min(max_units_per_plot, len(units))):
            # ランダムにユニットを選択
            if len(units) > 1:
                wi = np.random.randint(len(units))
            else:
                wi = 0
            
            unit = units[wi]
            nsp = len(unit.spikeTimeList)
            
            # サブプロットを作成
            ax = fig.add_subplot(grid[k//20, k%20])
            
            # 近くのサイトを見つける
            nearby_contacts = self.find_nearby_contacts(unit, contacts)
            if not nearby_contacts:
                ax.text(0.5, 0.5, 'No nearby contacts', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{nsp}', fontsize='small')
                ax.axis('off')
                continue
            
            # チャンネル数を制限
            n_chan = min(channels_per_unit, len(nearby_contacts))
            selected_contacts = nearby_contacts[:n_chan]
            
            # 各サイトの波形を描画
            amp = 4  # 振幅のスケール係数
            for ii, (contact, distance) in enumerate(selected_contacts):
                # スパイク波形を切り出し
                waveforms = self.extract_spike_waveforms(
                    contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                )
                
                if not waveforms:
                    continue
                
                # 平均波形を計算
                if len(waveforms) > 1:
                    wv = np.mean(waveforms, axis=0)
                else:
                    wv = waveforms[0]
                
                # 時間軸を作成
                t = np.arange(-len(wv)//2, len(wv)//2, 1, dtype='float32')
                t /= len(wv) / 20  # スケーリング
                
                # プローブ位置に波形を配置
                xi, yi = contact.x, contact.y
                ax.plot(xi + t, yi + wv * amp, lw=0.5, color='k', alpha=0.7)
            
            ax.set_title(f'{nsp}', fontsize='small')
            ax.axis('off')
        
        # 使用されていないサブプロットを非表示
        for k in range(len(units), max_units_per_plot):
            ax = fig.add_subplot(grid[k//20, k%20])
            ax.set_visible(False)
        
        plt.suptitle(f'All Unitss - {condition_name}', fontsize=14)
        
        # 保存
        if save_dir is not None:
            save_path = save_dir / f'probe_waveforms_{condition_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"プローブ波形プロットを保存しました: {save_path}")
        
        plt.show()
        
        return fig

class SpikeWaveformPlotter:
    """スパイク波形の切り出しと重ね書きを行うクラス"""
    
    def __init__(self, fs: float = 30000, window_ms: float = 4.0, 
                 max_distance: float = 200.0, max_waveforms: int = 100):
        """
        Args:
            fs: サンプリングレート (Hz)
            window_ms: 波形切り出しウィンドウ (ms)
            max_distance: 細胞から描画するサイトの最大距離 (μm)
            max_waveforms: 1つのサイトに描画する最大波形数
        """
        self.fs = fs
        self.window_ms = window_ms
        self.max_distance = max_distance
        self.max_waveforms = max_waveforms
        
        # 波形切り出しの設定
        self.window_size = int(window_ms / 1000 * fs)
        self.half_window = self.window_size // 2
        
        logging.info(f"波形切り出し設定: ウィンドウ={window_ms}ms, サンプル数={self.window_size}")
    
    def extract_spike_waveforms(self, signal: np.ndarray, spike_times: List[int], 
                               contact_id: int, unit_id: int) -> List[np.ndarray]:
        """指定されたサイトの信号からスパイク波形を切り出す"""
        waveforms = []
        
        for i, spike_time in enumerate(spike_times):
            if i >= self.max_waveforms:  # 最大波形数で制限
                break
                
            # スパイク時刻を中心とした波形切り出し
            start_idx = spike_time - self.half_window
            end_idx = start_idx + self.window_size
            
            # 範囲チェック
            if start_idx >= 0 and end_idx <= len(signal):
                waveform = signal[start_idx:end_idx]
                if len(waveform) == self.window_size:
                    waveforms.append(waveform)
            else:
                logging.debug(f"サイト{contact_id}, 細胞{unit_id}: スパイク{i}が範囲外 (時刻={spike_time}, 範囲=[{start_idx}:{end_idx}])")
        
        return waveforms
    
    def find_nearby_contacts(self, unit: Any, contacts: List[Any]) -> List[Tuple[Any, float]]:
        """細胞の近くにあるサイトを見つける"""
        nearby_contacts = []
        
        for contact in contacts:
            distance = calculate_distance_two_objects(unit, contact)
            if distance <= self.max_distance:
                nearby_contacts.append((contact, distance))
        
        # 距離順にソート
        nearby_contacts.sort(key=lambda x: x[1])
        
        logging.debug(f"細胞{unit.id}: 近くのサイト数={len(nearby_contacts)}, 最大距離={self.max_distance}μm")
        for contact, dist in nearby_contacts[:5]:  # 最初の5個のみ表示
            logging.debug(f"  サイト{contact.id}: 距離={dist:.1f}μm")
        
        return nearby_contacts
    
    def plot_unit_spike_waveforms(self, units: List[Any], contacts: List[Any], 
                                 condition_name: str = "", save_dir: Path = None) -> plt.Figure:
        """各細胞のスパイク波形を重ね書きしてプロット"""
        
        # プロット用の時間軸
        time_ms = np.linspace(-self.window_ms/2, self.window_ms/2, self.window_size)
        
        # 細胞ごとにプロット
        n_units = len(units)
        if n_units == 0:
            logging.warning("細胞が存在しません")
            return None
        
        # サブプロットのレイアウトを決定
        cols = min(3, n_units)  # 最大3列
        rows = (n_units + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        
        # axesの形状を適切に処理
        if n_units == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(rows, cols)
        
        # デバッグ情報
        logging.debug(f"サブプロットレイアウト: rows={rows}, cols={cols}")
        logging.debug(f"axes shape: {axes.shape}")
        logging.debug(f"axes type: {type(axes)}")
        
        fig.suptitle(f'Spike Waveforms - {condition_name}', fontsize=16)
        
        for unit_idx, unit in enumerate(units):
            if unit_idx >= rows * cols:
                break
                
            # 行と列のインデックスを計算
            row_idx = unit_idx // cols
            col_idx = unit_idx % cols
            
            # axesから正しいサブプロットを取得
            if rows == 1 and cols == 1:
                ax = axes[0]
            elif rows == 1:
                ax = axes[col_idx]
            elif cols == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            # 近くのサイトを見つける
            nearby_contacts = self.find_nearby_contacts(unit, contacts)
            
            if not nearby_contacts:
                ax.text(0.5, 0.5, f'Unit {unit.id}\nNo nearby contacts', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Unit {unit.id}')
                continue
            
            # 各サイトの波形を描画
            colors = plt.cm.viridis(np.linspace(0, 1, len(nearby_contacts)))
            
            for contact_idx, (contact, distance) in enumerate(nearby_contacts):
                # スパイク波形を切り出し
                waveforms = self.extract_spike_waveforms(
                    contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                )
                
                if not waveforms:
                    continue
                
                # 波形を重ね書き
                color = colors[contact_idx]
                alpha = 0.7
                
                for waveform in waveforms:
                    ax.plot(time_ms, waveform, color=color, alpha=alpha, linewidth=0.5)
                
                # 平均波形を太線で描画
                if len(waveforms) > 1:
                    mean_waveform = np.mean(waveforms, axis=0)
                    ax.plot(time_ms, mean_waveform, color=color, linewidth=2, 
                           label=f'Contact {contact.id} (d={distance:.0f}μm)')
                else:
                    ax.plot(time_ms, waveforms[0], color=color, linewidth=2,
                           label=f'Contact {contact.id} (d={distance:.0f}μm)')
            
            # 軸の設定
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title(f'Unit {unit.id} - {len(nearby_contacts)} nearby contacts')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Y軸の範囲を統一（オプション）
            ax.set_ylim(ax.get_ylim())
        
        # 使用されていないサブプロットを非表示
        for i in range(n_units, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            if rows == 1 and cols == 1:
                axes[0].set_visible(False)
            elif rows == 1:
                axes[col_idx].set_visible(False)
            elif cols == 1:
                axes[row_idx].set_visible(False)
            else:
                axes[row_idx, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存
        if save_dir is not None:
            save_path = save_dir / f'spike_waveforms_{condition_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"スパイク波形プロットを保存しました: {save_path}")
        
        return fig
    
    def plot_contact_spike_waveforms(self, units: List[Any], contacts: List[Any], 
                                 condition_name: str = "", save_dir: Path = None) -> plt.Figure:
        """各サイトのスパイク波形を重ね書きしてプロット"""
        
        # プロット用の時間軸
        time_ms = np.linspace(-self.window_ms/2, self.window_ms/2, self.window_size)
        
        # サイトごとにプロット
        n_contacts = len(contacts)
        if n_contacts == 0:
            logging.warning("サイトが存在しません")
            return None
        
        # サブプロットのレイアウトを決定
        cols = min(4, n_contacts)  # 最大4列
        rows = (n_contacts + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        
        # axesの形状を適切に処理
        if n_contacts == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes.reshape(rows, cols)
        
        # デバッグ情報
        logging.debug(f"サブプロットレイアウト: rows={rows}, cols={cols}")
        logging.debug(f"axes shape: {axes.shape}")
        logging.debug(f"axes type: {type(axes)}")
        
        fig.suptitle(f'Contact Spike Waveforms - {condition_name}', fontsize=16)
        
        for contact_idx, contact in enumerate(contacts):
            if contact_idx >= rows * cols:
                break
                
            # 行と列のインデックスを計算
            row_idx = contact_idx // cols
            col_idx = contact_idx % cols
            
            # axesから正しいサブプロットを取得
            if rows == 1 and cols == 1:
                ax = axes[0]
            elif rows == 1:
                ax = axes[col_idx]
            elif cols == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, col_idx]
            
            # 各細胞のスパイク波形を描画
            colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(units))))
            
            for unit_idx, unit in enumerate(units):
                if unit_idx >= 10:  # 最大10個の細胞まで
                    break
                    
                # スパイク波形を切り出し
                waveforms = self.extract_spike_waveforms(
                    contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                )
                
                if not waveforms:
                    continue
                
                # 波形を重ね書き
                color = colors[unit_idx % len(colors)]
                alpha = 0.6
                
                for waveform in waveforms:
                    ax.plot(time_ms, waveform, color=color, alpha=alpha, linewidth=0.3)
                
                # 平均波形を太線で描画
                if len(waveforms) > 1:
                    mean_waveform = np.mean(waveforms, axis=0)
                    ax.plot(time_ms, mean_waveform, color=color, linewidth=1.5,
                           label=f'Unit {unit.id}')
                else:
                    ax.plot(time_ms, waveforms[0], color=color, linewidth=1.5,
                           label=f'Unit {unit.id}')
            
            # 軸の設定
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title(f'Contact {contact.id}')
            ax.grid(True, alpha=0.3)
            
            # 凡例は最初のサイトのみ表示（スペース節約）
            if contact_idx == 0:
                ax.legend(fontsize=6, loc='upper right')
        
        # 使用されていないサブプロットを非表示
        for i in range(n_contacts, rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            if rows == 1 and cols == 1:
                axes[0].set_visible(False)
            elif rows == 1:
                axes[col_idx].set_visible(False)
            elif cols == 1:
                axes[row_idx].set_visible(False)
            else:
                axes[row_idx, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        # 保存
        if save_dir is not None:
            save_path = save_dir / f'contact_waveforms_{condition_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"サイト波形プロットを保存しました: {save_path}")
        
        return fig
    
    def plot_waveform_statistics(self, units: List[Any], contacts: List[Any], 
                                condition_name: str = "", save_dir: Path = None) -> plt.Figure:
        """波形の統計情報をプロット"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Waveform Statistics - {condition_name}', fontsize=16)
        
        # 1. 各サイトでの検出スパイク数
        ax1 = axes[0, 0]
        contact_spike_counts = []
        contact_ids = []
        
        for contact in contacts:
            total_spikes = 0
            for unit in units:
                waveforms = self.extract_spike_waveforms(
                    contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                )
                total_spikes += len(waveforms)
            
            contact_spike_counts.append(total_spikes)
            contact_ids.append(contact.id)
        
        ax1.bar(contact_ids, contact_spike_counts, alpha=0.7)
        ax1.set_xlabel('Contact ID')
        ax1.set_ylabel('Total Spikes Detected')
        ax1.set_title('Spikes per Contact')
        ax1.grid(True, alpha=0.3)
        
        # 2. 各細胞の総スパイク数
        ax2 = axes[0, 1]
        unit_spike_counts = [len(unit.spikeTimeList) for unit in units]
        unit_ids = [unit.id for unit in units]
        
        ax2.bar(unit_ids, unit_spike_counts, alpha=0.7)
        ax2.set_xlabel('Unit ID')
        ax2.set_ylabel('Total Spikes')
        ax2.set_title('Spikes per Unit')
        ax2.grid(True, alpha=0.3)
        
        # 3. 距離vs検出スパイク数の関係
        ax3 = axes[1, 0]
        distances = []
        detected_spikes = []
        
        for unit in units:
            for contact in contacts:
                distance = calculate_distance_two_objects(unit, contact)
                if distance <= self.max_distance:
                    waveforms = self.extract_spike_waveforms(
                        contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                    )
                    distances.append(distance)
                    detected_spikes.append(len(waveforms))
        
        ax3.scatter(distances, detected_spikes, alpha=0.6)
        ax3.set_xlabel('Distance (μm)')
        ax3.set_ylabel('Detected Spikes')
        ax3.set_title('Distance vs Detection')
        ax3.grid(True, alpha=0.3)
        
        # 4. 波形振幅の分布
        ax4 = axes[1, 1]
        all_amplitudes = []
        
        for unit in units:
            for contact in contacts:
                waveforms = self.extract_spike_waveforms(
                    contact.signalRaw, unit.spikeTimeList, contact.id, unit.id
                )
                for waveform in waveforms:
                    amplitude = np.max(np.abs(waveform))
                    all_amplitudes.append(amplitude)
        
        if all_amplitudes:
            ax4.hist(all_amplitudes, bins=30, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Peak Amplitude (μV)')
            ax4.set_ylabel('Count')
            ax4.set_title('Amplitude Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        if save_dir is not None:
            save_path = save_dir / f'waveform_statistics_{condition_name}.png'
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"波形統計プロットを保存しました: {save_path}")
        
        return fig

def plot_main(units: List[Any], contacts: List[Any], condition_name: str = "", 
              save_dir: Path = None) -> Dict[str, plt.Figure]:
    """メインのプロット関数"""
    
    logging.info(f"スパイク波形プロット開始: 細胞数={len(units)}, サイト数={len(contacts)}")
    
    # プロッターの初期化
    plotter = SpikeWaveformPlotter()
    template_visualizer = TemplateWaveformVisualizer()
    
    figures = {}
    
    try:
        # # 1. 細胞ごとのスパイク波形
        # logging.info("細胞ごとのスパイク波形をプロット中...")
        # unit_fig = plotter.plot_unit_spike_waveforms(units, contacts, condition_name, save_dir)
        # if unit_fig:
        #     figures['unit_waveforms'] = unit_fig
        
        # # 2. サイトごとのスパイク波形
        # logging.info("サイトごとのスパイク波形をプロット中...")
        # contact_fig = plotter.plot_contact_spike_waveforms(units, contacts, condition_name, save_dir)
        # if contact_fig:
        #     figures['contact_waveforms'] = contact_fig
        
        # # 3. 波形統計
        # logging.info("波形統計をプロット中...")
        # stats_fig = plotter.plot_waveform_statistics(units, contacts, condition_name, save_dir)
        # if stats_fig:
        #     figures['statistics'] = stats_fig
        
        # 4. テンプレートベースのプローブ位置波形可視化
        logging.info("テンプレートベースのプローブ位置波形可視化中...")
        probe_fig = template_visualizer.plot_probe_waveforms(units, contacts, condition_name, save_dir)
        if probe_fig:
            figures['probe_waveforms'] = probe_fig
        
        # # 5. 細胞波形のプローブ配置表示
        # logging.info("細胞波形のプローブ配置表示中...")
        # unit_probe_fig = template_visualizer.plot_unit_waveforms_on_probe(units, contacts, condition_name, save_dir)
        # if unit_probe_fig:
        #     figures['unit_waveforms_on_probe'] = unit_probe_fig
        
 
        plt.show()
        
        logging.info("スパイク波形プロット完了")
        
    except Exception as e:
        logging.error(f"スパイク波形プロットでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    return figures

if __name__ == "__main__":
    # テスト用
    logging.basicConfig(level=logging.INFO)
    print("このモジュールは直接実行せず、importして使用してください。")
