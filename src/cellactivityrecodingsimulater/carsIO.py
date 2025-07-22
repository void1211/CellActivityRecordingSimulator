import json
import numpy as np
from pathlib import Path
import chardet

from pydantic import BaseModel

from Site import Site
from Cell import Cell
from Settings import Settings

def load_settings(path: str) -> Settings:
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
            content = f.read()
            print(f"ファイル内容: {repr(content)}")
            if not content.strip():
                raise ValueError(f"ファイルが空です: {path}")
            return Settings(**json.loads(content))
    except json.JSONDecodeError as e:
        print(f"JSONデコードエラー: {e}")
        print(f"ファイル内容: {repr(content)}")
        raise
    except Exception as e:
        print(f"その他のエラー: {e}")
        raise
    
def load_cells(path: Path) -> list[Cell]:
    cells = []
    # ファイルのエンコーディングを自動検出
    with open(path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
    
    with open(path, "r", encoding=encoding) as f:
        jcells = json.load(f)
    for i in range(len(jcells["id"])):
        cells.append(Cell(id=jcells["id"][i], 
                          x=jcells["x"][i], 
                          y=jcells["y"][i], 
                          z=jcells["z"][i]))
    return cells
    
def load_sites(path: Path) -> list[Site]:
    sites = []
    # ファイルのエンコーディングを自動検出
    with open(path, "rb") as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']
    
    with open(path, "r", encoding=encoding) as f:
        jsites = json.load(f)
    for i in range(len(jsites["id"])):
        sites.append(Site(id=jsites["id"][i], 
                          x=jsites["x"][i], 
                          y=jsites["y"][i],
                          z=jsites["z"][i]))
    return sites

def load_spikeTemplates(path: Path) -> list[np.ndarray]:
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

def save_data(path: Path, cells: list[Cell], sites: list[Site]):

    # todo ファイル名を設定できるようにする
    # todo 必要事項全て保存できるようにする
    
    signalRaw = np.array([site.signalRaw for site in sites])
    signalNoise = np.array([site.signalNoise for site in sites])
    signalFiltered = np.array([site.signalFiltered for site in sites])
    
    # 異なる長さのリストをobject型で保存
    spikeTimeList = np.array([cell.spikeTimeList for cell in cells], dtype=object)
    spikeAmpList = np.array([cell.spikeAmpList for cell in cells], dtype=object)
    spikeTemp = np.array([cell.spikeTemp for cell in cells], dtype=object)

    # 浮動小数点データをint16に変換
    signalRaw_int16 = signalRaw.astype(np.int16)
    signalNoise_int16 = signalNoise.astype(np.int16)
    signalFiltered_int16 = signalFiltered.astype(np.int16)
    
    # int16データを.npyファイルとして保存
    np.save(path / "signalRaw.npy", signalRaw_int16)
    np.save(path / "signalNoise.npy", signalNoise_int16)
    np.save(path / "signalFiltered.npy", signalFiltered_int16)
    
    # バイナリファイルに保存（int16）
    with open(path / "signalRaw.bin", "wb") as f:
        signalRaw_int16.reshape((1,-1), order="F").tofile(f)
    with open(path / "signalNoise.bin", "wb") as f:
        signalNoise_int16.reshape((1,-1), order="F").tofile(f)
    with open(path / "signalFiltered.bin", "wb") as f:
        signalFiltered_int16.reshape((1,-1), order="F").tofile(f)

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

def convert_sites_for_kilosort(sites: list[Site]) -> dict:
    """
    sitesをKilosort用のprobe形式に変換する
    """
    # チャンネルマップを作成
    chanMap = [site.id for site in sites]
    if min(chanMap) != 0:
        chanMap = [chanMap[i] - min(chanMap) for i in range(len(chanMap))]
    # 座標を抽出
    xc = [site.x for site in sites]
    yc = [site.y for site in sites]
    
    # kcoords（プローブグループ）を設定（デフォルトは全て1）
    kcoords = [0] * len(sites)
    
    # チャンネル数を設定
    n_chan = len(sites)
    
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
    with open(path / "KS_probe.json", "w") as f:
        json.dump(probe, f, indent=2)
