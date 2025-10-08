import json
import numpy as np
from pathlib import Path, WindowsPath
import chardet
import logging
import os

from .Contact import Contact
from .Unit import Unit
from probeinterface import Probe

# Windowsの場合はWindowsPathを使用
if os.name == 'nt':
    Path = WindowsPath

def load_settings_from_json(path: str):
    from .Settings import Settings
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
                from .Settings import convert_legacySettings
                content = convert_legacySettings(content)
            settings = load_settings_from_dict(content)
            logging.info(f"設定ファイル: {settings}")
            return settings
    except json.JSONDecodeError as e:
        logging.error(f"JSONデコードエラー: {e}")
        logging.error(f"ファイル内容: {repr(content)}")
        raise
    except Exception as e:
        logging.error(f"その他のエラー: {e}")
        raise

def load_settings_from_dict(data: dict) -> "Settings":
    from .Settings import Settings
    return Settings.from_dict(data)

def load_units_from_json(object: Path|dict) -> list[Unit]:
    if isinstance(object, Path):
        if not Path(object).exists():
            raise FileNotFoundError(f"セルファイルが見つかりません: {object}")
        
        if Path(object).stat().st_size == 0:
            raise ValueError(f"セルファイルが空です: {object}")

    units = []

    if isinstance(object, dict):
        junits = object
    else:
        with open(object, "r") as f:
            junits = json.load(f)

    for i in range(len(junits["id"])):
        unit_data = {
            "id": junits["id"][i], 
            "x": junits["x"][i], 
            "y": junits["y"][i], 
            "z": junits["z"][i]
        }
        units.append(Unit.from_dict(unit_data)) 
    return units

def load_units_from_dict(data: dict) -> list[Unit]:
    from .Unit import Unit
    return [Unit.from_dict(unit_data) for unit_data in data]


def load_contacts_from_json(object: Path|dict) -> list[Contact]:
    if isinstance(object, Path):
        if not Path(object).exists():
            raise FileNotFoundError(f"サイトファイルが見つかりません: {object}")
        
        if Path(object).stat().st_size == 0:
            raise ValueError(f"サイトファイルが空です: {object}")

    contacts = []
    if isinstance(object, dict):
        jcontacts = object
    else:
        with open(object, "r") as f:
            jcontacts = json.load(f)

    for i in range(len(jcontacts["id"])):
        contact_data = {
            "id": jcontacts["id"][i], 
            "x": jcontacts["x"][i], 
            "y": jcontacts["y"][i], 
            "z": jcontacts["z"][i]
        }
        contacts.append(Contact.from_dict(contact_data))
    return contacts

def load_contacts_from_Probe(probeObject) -> list[Contact]:
    contacts = []
    if not isinstance(probeObject, Probe):
        raise ValueError(f"Invalid probe type")
    probe_dict = probeObject.to_dict()
    contact_positions = np.array(probe_dict["contact_positions"])
    is_3d = True
    if contact_positions.shape[1] != 3:
        is_3d = False
    for i in range(len(contact_positions)):
        if is_3d:
            contact_data = {
                "id": i,
                "x": contact_positions[i][0],
                "y": contact_positions[i][1],
                "z": contact_positions[i][2]
            }
        else:
            contact_data = {
                "id": i,
                "x": contact_positions[i][0],
                "y": contact_positions[i][1],
                "z": 0
            }
        contacts.append(Contact.from_dict(contact_data))
    return contacts

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

def save_data(path: Path, units: list[Unit], contacts: list[Contact], noise_units: list[Unit]=None, fs: float=None):
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

    signalRaw = np.array([contact.get_signal("raw") for contact in contacts])
    signalNoise = np.array([contact.get_signal("noise") for contact in contacts])
    signalFiltered = np.array([contact.get_signal("filtered", fs=fs) for contact in contacts])
    signalPowerNoise = np.array([contact.get_signal("power") for contact in contacts])
    signalDrift = np.array([contact.get_signal("drift") for contact in contacts])
    signalBGNoise = np.array([contact.get_signal("background") for contact in contacts])
    signalSpike = np.array([contact.get_signal("spike") for contact in contacts])

    # グループ情報を保存
    group = np.array([unit.group for unit in units], dtype=object)

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
            signalRaw_int16.reshape((1, -1), order="F").tofile(f)
        with open(str(path / "signalNoise.bin"), "wb") as f:
            signalNoise_int16.reshape((1, -1), order="F").tofile(f)
        with open(str(path / "signalFiltered.bin"), "wb") as f:
            signalFiltered_int16.reshape((1, -1), order="F").tofile(f)
        with open(str(path / "signalPowerNoise.bin"), "wb") as f:
            signalPowerNoise_int16.reshape((1, -1), order="F").tofile(f)
        with open(str(path / "signalDrift.bin"), "wb") as f:
            signalDrift_int16.reshape((1, -1), order="F").tofile(f)
        with open(str(path / "signalBGNoise.bin"), "wb") as f:
            signalBGNoise_int16.reshape((1, -1), order="F").tofile(f)
        with open(str(path / "signalSpike.bin"), "wb") as f:
            signalSpike_int16.reshape((1, -1), order="F").tofile(f)
    except PermissionError as e:
        logging.error(f"ファイルが他のプログラムで開かれています: {e}")
        logging.error("ファイルを閉じてから再実行してください")
        raise
    except OSError as e:
        logging.error(f"ファイル保存でエラーが発生しました: {e}")
        raise

    unit_ids = []
    unit_positions = []
    spike_times = []
    spike_amps = []
    spike_temps = []
    for unit in units:
        p = [unit.x, unit.y, unit.z]
        unit_ids.append(unit.id)
        unit_positions.append(p)
        spike_times.append(unit.spikeTimeList)
        spike_amps.append(unit.spikeAmpList)
        spike_temps.append(unit.spikeTemp.template)

    unit_ids = np.array(unit_ids)
    unit_positions = np.array(unit_positions)
    spike_times = np.array(spike_times, dtype=object)
    spike_amps = np.array(spike_amps, dtype=object)
    spike_temps = np.array(spike_temps, dtype=object)

    contact_ids = []
    contact_positions = []
    for contact in contacts:
        p = [contact.x, contact.y, contact.z]
        contact_ids.append(contact.id)
        contact_positions.append(p)

    contact_ids = np.array(contact_ids)
    contact_positions = np.array(contact_positions)

    # 配列はnp.saveで保存（形状とデータ型を保持）
    np.save(path / "unit_ids.npy", unit_ids)
    np.save(path / "unit_positions.npy", unit_positions)
    np.save(path / "contact_ids.npy", contact_ids)
    np.save(path / "contact_positions.npy", contact_positions)

    # object型の配列はnp.saveで保存
    np.save(path / "spike_times.npy", spike_times)
    np.save(path / "spike_amplitudes.npy", spike_amps)
    np.save(path / "spike_templates.npy", spike_temps)

    # probe形式でcontactsを保存
    save_probe_data(path, contacts)

    if noise_units is not None:
        noise_units_ids = []
        noise_units_positions = []
        noise_units_spike_times = []
        noise_units_spike_amps = []
        noise_units_spike_temps = []
        for noise_unit in noise_units:
            p = [noise_unit.x, noise_unit.y, noise_unit.z]

            noise_units_ids.append(noise_unit.id)
            noise_units_positions.append(p)
            noise_units_spike_times.append(noise_unit.spikeTimeList)
            noise_units_spike_amps.append(noise_unit.spikeAmpList)
            noise_units_spike_temps.append(noise_unit.spikeTemp.template)

        noise_unit_ids = np.array(noise_units_ids)
        noise_unit_positions = np.array(noise_units_positions)
        noise_spike_times = np.array(noise_units_spike_times, dtype=object)
        noise_spike_amps = np.array(noise_units_spike_amps, dtype=object)
        noise_spike_temps = np.array(noise_units_spike_temps, dtype=object)

        np.save(path / "noise_unit_ids.npy", noise_unit_ids)
        np.save(path / "noise_unit_positions.npy", noise_unit_positions)
        np.save(path / "noise_unit_spike_times.npy", noise_spike_times)
        np.save(path / "noise_unit_spike_amplitudes.npy", noise_spike_amps)
        np.save(path / "noise_unit_spike_templates.npy", noise_spike_temps)


def convert_contacts_for_kilosort(contacts: list[Contact]) -> dict:
    """
    contactsをKilosort用のprobe形式に変換する
    """
    # チャンネルマップを作成
    chanMap = [int(contact.id) for contact in contacts]
    if min(chanMap) != 0:
        chanMap = [chanMap[i] - min(chanMap) for i in range(len(chanMap))]
    # 座標を抽出
    xc = [float(contact.x) for contact in contacts]
    yc = [float(contact.y) for contact in contacts]

    # kcoords（プローブグループ）を設定（デフォルトは全て1）
    kcoords = [0] * len(contacts)

    # チャンネル数を設定
    n_chan = int(len(contacts))

    probe = {
        'chanMap': chanMap,
        'xc': xc,
        'yc': yc,
        'kcoords': kcoords,
        'n_chan': n_chan
    }
    return probe


def save_probe_data(path: Path, contacts: list[Contact]):
    """
    contactsをKilosort用のprobe形式で保存する
    """
    probe = convert_contacts_for_kilosort(contacts)
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
