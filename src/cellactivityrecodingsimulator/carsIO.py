import json
import numpy as np
from pathlib import Path, WindowsPath
import chardet
import logging
import os

from pydantic import BaseModel

from .Site import Site
from .Cell import Cell
from .Settings import Settings
from probeinterface import Probe
from .tools import convert_legacySettings

# Windowsの場合はWindowsPathを使用
if os.name == 'nt':
    Path = WindowsPath

def load_settings_file(path: str) -> Settings:
    # ファイルの存在確認
    if not Path(path).exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    
    # ファイルサイズの確認
    file_size = Path(path).stat().st_size
    if file_size == 0:
        raise ValueError(f"設定ファイルが空です: {path}")
    
    # ファイルのエンコーディングを自動検出
    with open(path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
    
    try:
        with open(path, "r", encoding=encoding) as f:
            content = json.load(f)
            # logging.info(f"ファイル内容: {repr(content)}")
            if "baseSettings" not in content:
                content = convert_legacySettings(content)
            settings = Settings(content)
            logging.info(f"設定ファイル: {settings}")
            # # 設定の検証を実行
            # validation_summary = settings.get_validation_summary()
            # logging.info(f"設定検証結果: {validation_summary}")
            
            # # エラーがある場合は警告を出力（実行は継続）
            # errors = settings.validate_settings()
            # if errors:
            #     logging.warning(f"設定に{len(errors)}個の警告がありますが、処理を継続します:")
            #     for error in errors:
            #         logging.warning(f"  - {error}")
            
            return settings
    except json.JSONDecodeError as e:
        logging.error(f"JSONデコードエラー: {e}")
        logging.error(f"ファイル内容: {repr(content)}")
        raise
    except Exception as e:
        logging.error(f"その他のエラー: {e}")
        raise
    
def load_cells_from_json(path: Path) -> list[Cell]:

    if not Path(path).exists():
        raise FileNotFoundError(f"セルファイルが見つかりません: {path}")
    
    if Path(path).stat().st_size == 0:
        raise ValueError(f"セルファイルが空です: {path}")
    
    cells = []
    # ファイルのエンコーディングを自動検出
    with open(path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
    
    with open(path, "r", encoding=encoding) as f:
        jcells = json.load(f)
    
    for i in range(len(jcells["id"])):
        cell_data = {
            "id": jcells["id"][i], 
            "x": jcells["x"][i], 
            "y": jcells["y"][i], 
            "z": jcells["z"][i]
        }
        
        cells.append(Cell().from_dict(cell_data))
    
    return cells
    
def load_sites_from_json(path: Path) -> list[Site]:

    if not Path(path).exists():
        raise FileNotFoundError(f"サイトファイルが見つかりません: {path}")
    
    if Path(path).stat().st_size == 0:
        raise ValueError(f"サイトファイルが空です: {path}")
    
    sites = []
    # ファイルのエンコーディングを自動検出
    with open(path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
    
    with open(path, "r", encoding=encoding) as f:
        jsites = json.load(f)
    for i in range(len(jsites["id"])):
        site_data = {
            "id": jsites["id"][i], 
            "x": jsites["x"][i], 
            "y": jsites["y"][i], 
            "z": jsites["z"][i]
        }
        sites.append(Site().from_dict(site_data))
    return sites

def load_sites_from_Probe(probeObject) -> list[Site]:
    sites = []
    if not isinstance(probeObject, Probe):
        raise ValueError(f"Invalid probe type")
    probe_dict = probeObject.to_dict()
    site_positions = np.array(probe_dict["contact_positions"])
    is_3d = True
    if site_positions.shape[1] != 3:
        is_3d = False
    for i in range(len(site_positions)):
        if is_3d:
            site_data = {
                "id": i,
                "x": site_positions[i][0],
                "y": site_positions[i][1],
                "z": site_positions[i][2]
            }
        else:
            site_data = {
                "id": i,
                "x": site_positions[i][0],
                "y": site_positions[i][1],
                "z": 0
            }
        sites.append(Site().from_dict(site_data))
    return sites

def load_spike_templates(path: Path) -> list[np.ndarray]:
    spikeTemplates = []
    # ファイルのエンコーディングを自動検出
    with open(path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
    
    with open(path, "r", encoding=encoding) as f:
        jspikeTemplates = json.load(f)
    for i in range(len(jspikeTemplates["id"])):
        spikeTemplates.append(np.array(jspikeTemplates["spikeTemplate"][i]))
    return spikeTemplates

def load_noise_file(path: Path) -> np.ndarray:
    """真の録音ノイズを取得する"""
    
    try:
        noise = np.load(path).astype(np.float64)
        return noise
    except FileNotFoundError:
        logging.warning(f"File not found: {path}")
        return None
    

def save_data(path: Path, cells: list[Cell], sites: list[Site], noise_cells: list[Cell]=None, fs: float=None):
    # パラメータの検証
    if not isinstance(path, Path):
        raise TypeError(f"path must be a Path object, got {type(path)}")
    
    logging.info(f"データ保存開始: {path}")
    
    # ディレクトリが存在することを確認
    if not path.exists():
        logging.warning(f"保存先ディレクトリが存在しません: {path}")
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"保存先ディレクトリを作成しました: {path}")
    
    # todo ファイル名を設定できるようにする
    # todo 必要事項全て保存できるようにする
    
    signalRaw = np.array([site.get_signal("raw") for site in sites])
    signalNoise = np.array([site.get_signal("noise") for site in sites])
    signalFiltered = np.array([site.get_signal("filtered", fs=fs) for site in sites])
    signalPowerNoise = np.array([site.get_signal("power") for site in sites])
    signalDrift = np.array([site.get_signal("drift") for site in sites])
    signalBGNoise = np.array([site.get_signal("background") for site in sites])
    signalSpike = np.array([site.get_signal("spike") for site in sites])

    # グループ情報を保存
    group = np.array([cell.group for cell in cells], dtype=object)
    # 異なる長さのリストをobject型で保存
    spikeTimeList = np.array([cell.spikeTimeList for cell in cells], dtype=object)
    spikeAmpList = np.array([cell.spikeAmpList for cell in cells], dtype=object)
    spikeTemp = np.array([cell.spikeTemp for cell in cells], dtype=object)

    # 浮動小数点データをint16に変換
    signalRaw_int16 = signalRaw.astype(np.int16)
    signalNoise_int16 = signalNoise.astype(np.int16)
    signalFiltered_int16 = signalFiltered.astype(np.int16)
    signalPowerNoise_int16 = signalPowerNoise.astype(np.int16)
    signalDrift_int16 = signalDrift.astype(np.int16)
    signalBGNoise_int16 = signalBGNoise.astype(np.int16)
    signalSpike_int16 = signalSpike.astype(np.int16)
    # int16データを.npyファイルとして保存
    np.save(path / "signalRaw.npy", signalRaw_int16)
    np.save(path / "signalNoise.npy", signalNoise_int16)
    np.save(path / "signalFiltered.npy", signalFiltered_int16)
    np.save(path / "signalPowerNoise.npy", signalPowerNoise_int16)
    np.save(path / "signalDrift.npy", signalDrift_int16)
    np.save(path / "signalBGNoise.npy", signalBGNoise_int16)
    np.save(path / "signalSpike.npy", signalSpike_int16)
    np.save(path / "group.npy", group)
    
    # バイナリファイルに保存（int16）
    try:
        with open(str(path / "signalRaw.bin"), "wb") as f:
            signalRaw_int16.reshape((1,-1), order="F").tofile(f)
        with open(str(path / "signalNoise.bin"), "wb") as f:
            signalNoise_int16.reshape((1,-1), order="F").tofile(f)
        with open(str(path / "signalFiltered.bin"), "wb") as f:
            signalFiltered_int16.reshape((1,-1), order="F").tofile(f)
        with open(str(path / "signalPowerNoise.bin"), "wb") as f:
            signalPowerNoise_int16.reshape((1,-1), order="F").tofile(f)
        with open(str(path / "signalDrift.bin"), "wb") as f:
            signalDrift_int16.reshape((1,-1), order="F").tofile(f)
        with open(str(path / "signalBGNoise.bin"), "wb") as f:
            signalBGNoise_int16.reshape((1,-1), order="F").tofile(f)
        with open(str(path / "signalSpike.bin"), "wb") as f:
            signalSpike_int16.reshape((1,-1), order="F").tofile(f)
    except PermissionError as e:
        logging.error(f"ファイルが他のプログラムで開かれています: {e}")
        logging.error("ファイルを閉じてから再実行してください")
        raise
    except OSError as e:
        logging.error(f"ファイル保存でエラーが発生しました: {e}")
        raise
    
    cell_ids = np.array([cell.id for cell in cells])
    cell_positions = np.array([[cell.x, cell.y, cell.z] for cell in cells])
    spike_times = np.array([cell.spikeTimeList for cell in cells], dtype=object)
    spike_amps = np.array([cell.spikeAmpList for cell in cells], dtype=object)
    spike_temps = np.array([cell.spikeTemp for cell in cells], dtype=object)
    
    site_ids = np.array([site.id for site in sites])
    site_positions = np.array([[site.x, site.y, site.z] for site in sites])

    # 配列はnp.saveで保存（形状とデータ型を保持）
    np.save(path / "cell_ids.npy", cell_ids)
    np.save(path / "cell_positions.npy", cell_positions)
    np.save(path / "site_ids.npy", site_ids)
    np.save(path / "site_positions.npy", site_positions)
    
    # object型の配列はnp.saveで保存
    np.save(path / "spike_times.npy", spike_times)
    np.save(path / "spike_amplitudes.npy", spike_amps)
    np.save(path / "spike_templates.npy", spike_temps)
    
    # probe形式でsitesを保存
    save_probe_data(path, sites)

    if noise_cells is not None:
        noise_cell_ids = np.array([noise_cell.id for noise_cell in noise_cells])
        noise_cell_positions = np.array([[noise_cell.x, noise_cell.y, noise_cell.z] for noise_cell in noise_cells])
        noise_spike_times = np.array([noise_cell.spikeTimeList for noise_cell in noise_cells], dtype=object)
        noise_spike_amps = np.array([noise_cell.spikeAmpList for noise_cell in noise_cells], dtype=object)
        noise_spike_temps = np.array([noise_cell.spikeTemp for noise_cell in noise_cells], dtype=object)
        
        np.save(path / "noise_cell_ids.npy", noise_cell_ids)
        np.save(path / "noise_cell_positions.npy", noise_cell_positions)
        np.save(path / "noise_cell_spike_times.npy", noise_spike_times)
        np.save(path / "noise_cell_spike_amplitudes.npy", noise_spike_amps)
        np.save(path / "noise_cell_spike_templates.npy", noise_spike_temps)

def convert_sites_for_kilosort(sites: list[Site]) -> dict:
    """
    sitesをKilosort用のprobe形式に変換する
    """
    # チャンネルマップを作成
    chanMap = [int(site.id) for site in sites]
    if min(chanMap) != 0:
        chanMap = [chanMap[i] - min(chanMap) for i in range(len(chanMap))]
    # 座標を抽出
    xc = [float(site.x) for site in sites]
    yc = [float(site.y) for site in sites]
    
    # kcoords（プローブグループ）を設定（デフォルトは全て1）
    kcoords = [0] * len(sites)
    
    # チャンネル数を設定
    n_chan = int(len(sites))
    
    probe = {
        'chanMap': chanMap,
        'xc': xc,
        'yc': yc,
        'kcoords': kcoords,
        'n_chan': n_chan
    }
    
    return probe

def save_probe_data(path: Path, sites: list[Site]):
    """
    sitesをKilosort用のprobe形式で保存する
    """
    probe = convert_sites_for_kilosort(sites)
    
    # JSON形式で保存
    try:
        with open(str(path / "KS_probe.json"), "w") as f:
            json.dump(probe, f, indent=2)
    except PermissionError as e:
        logging.error(f"KS_probe.jsonファイルが他のプログラムで開かれています: {e}")
        logging.error("ファイルを閉じてから再実行してください")
        raise
    except OSError as e:
        logging.error(f"KS_probe.jsonファイル保存でエラーが発生しました: {e}")
        raise

#=============test=============

class TestLoadCells():

    def test_classinit(self):
        cell_data ={
            "id": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "group": 0
        }   
        cell1 = Cell(**cell_data)
        print("cell(data):", cell1.__repr__() == "Cell(id=0, x=0, y=0, z=0, group=0)")

        cell_data ={
        "id": 0,
        "x": 0,
        "y": 0,
        "z": 0,
        }
        cell2 = Cell(**cell_data)
        print("cell(data):", cell2.__repr__() == "Cell(id=0, x=0, y=0, z=0, group=0)")

    def test_from_dict(self):
        cell_data ={
            "id": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "group": 0
        }  

        cell1 = Cell().from_dict(cell_data)
        print("Cell.from_dict(data):", cell1.__repr__() == "Cell(id=0, x=0, y=0, z=0, group=0)")

        cell_data ={
            "id": 0,
            "x": 0,
            "y": 0,
            "z": 0,
        }
        cell2 = Cell().from_dict(cell_data)
        print("Cell.from_dict(data):", cell2.__repr__() == "Cell(id=0, x=0, y=0, z=0, group=0)")

class TestLoadSites():

    def test_classinit(self):
        site_data ={
            "id": 0,
            "x": 0,
            "y": 0,
            "z": 0,
        }
        site1 = Site(**site_data)
        print("site(data):", site1.__repr__() == "Site(id=0, x=0, y=0, z=0)")

    def test_from_dict(self):
        site_data ={
            "id": 0,
            "x": 0,
            "y": 0,
            "z": 0,
        }
        site1 = Site().from_dict(site_data)
        print("Site.from_dict(data):", site1.__repr__() == "Site(id=0, x=0, y=0, z=0)")
  