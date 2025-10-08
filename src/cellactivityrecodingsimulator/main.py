import numpy as np
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
from probeinterface import Probe

from .GroundTruthUnitsObject import GTUnitsObject
from .BackGroundUnitsObject import BGUnitsObject
from .CarsObject import CarsObject
from .Settings import Settings
from .carsIO import load_settings_from_json, load_units_from_json, load_contacts_from_Probe, load_contacts_from_json, save_data, load_noise_file, load_spike_templates
from .Noise import RandomNoise, DriftNoise, PowerLineNoise
from .Template import BaseTemplate
from .calculate import calculate_scaled_spike_amplitude, calculate_distance_two_objects
from .tools import addSpikeToSignal, make_save_dir
from .plot.main import plot_main
from .ProbeObject import ProbeObject
# ベースディレクトリを固定
BASE_DIR = Path("simulations")
TEST_DIR = Path("test")
SETTINGS_FILE = Path("settings.json")
UNITS_FILE = Path("cells.json")
PROBE_FILE = Path("probe.json")

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def init_run(settings: Settings, dir: Path, verbose):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    baseSettings = settings.rootSettings.to_dict()
    # 保存ディレクトリの作成
    if baseSettings["pathSaveDir"] is None:
        pathSaveDir = dir
    else:
        pathSaveDir = baseSettings["pathSaveDir"]
        
    saveDir = make_save_dir(pathSaveDir)
    return saveDir

def load_files(object: Path|Probe, type: str):
    if type == "settings":
        return load_settings_from_json(object)
    elif type == "units":
        return load_units_from_json(object)
    elif type == "probe" and isinstance(object, Path):
        return load_contacts_from_json(object)
    elif type == "probe" and isinstance(object, Probe):
        return load_contacts_from_Probe(object)
    else:
        raise ValueError(f"Invalid file type")

def run(
    dir: Path, 
    settings:Path|Settings|dict|None=None, 
    units:Path|GTUnitsObject|dict|None=None, 
    probe:Path|Probe|None=None, 
    plot: bool=False, 
    verbose: bool=False
    ):
    """単一の実験を実行する"""
    try:
        print(type(settings))
        print(type(units))
        print(type(probe))
        logging.info(f"=== 実験開始 ===")
        settings = Settings.load(settings)
        logging.info(f"設定ファイル読み込み完了")
        
        # セルデータの読み込み
        gt_units = GTUnitsObject.load(units, settings=settings.to_dict())
        logging.info(f"セルデータ読み込み完了: {gt_units.get_units_num()} units")
        
        # サイトデータの読み込み
        probe = ProbeObject.load(probe)
        logging.info(f"サイトデータ読み込み完了: {probe.get_contacts_num()} contacts")

        saveDir = init_run(settings, Path(dir), verbose) 

        settings = settings.to_dict()
        # ノイズの適用
        logging.info(f"=== ノイズ生成と記録点への適用 ===")
        if settings["noiseSettings"]["noiseType"] == "truth":
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                noise = load_noise_file(settings["noiseSettings"]["truth"]["pathNoise"])
                contact.set_signal("background", noise)

        elif settings["noiseSettings"]["noiseType"] == "normal":  
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                noise = RandomNoise.generate("normal", settings)
                contact.set_signal("background", noise)

        elif settings["noiseSettings"]["noiseType"] == "gaussian":
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                noise = RandomNoise.generate("gaussian", settings)
                contact.set_signal("background", noise)

        elif settings["noiseSettings"]["noiseType"] == "model":
            # ノイズ細胞を生成してサイトに追加
            bg_units = BGUnitsObject.generate(settings, probe.contacts)
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                
                noise = bg_units.make_background_activity(contact, settings["spikeSettings"]["attenTime"], settings)
                contact.set_signal("background", noise)

        elif settings["noiseSettings"]["noiseType"] == "none":
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                noise = np.zeros(int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"]))
                contact.set_signal("background", noise)

        else:
            raise ValueError(f"Invalid noise type")
        
        # スパイクテンプレートの生成
        logging.info(f"=== スパイク活動の生成 ===")
        
        def is_valid_template(unit):
            """ユニットのテンプレートが有効かどうかをチェック"""
            return (
                hasattr(unit, 'spikeTemp') and 
                unit.spikeTemp is not None and
                hasattr(unit.spikeTemp, 'template') and
                len(unit.spikeTemp.template) > 1
            )
        
        if settings["spikeSettings"]["spikeType"] in ["gabor", "exponential"]:
            if settings["templateSimilarityControlSettings"]["enable"]:
                # 類似度制御が有効な場合：グループごとに類似テンプレートを生成
                for group_id in set(unit.group for unit in gt_units.units):
                    group_units = [u for u in gt_units.units if u.group == group_id]
                    base_template = None
                    
                    for unit in group_units:
                        if base_template is None:
                            if not is_valid_template(unit):
                                unit.spikeTemp = BaseTemplate.generate(settings)
                            base_template = unit.spikeTemp
                        else:
                            unit.spikeTemp = base_template.generate_similar_template(settings)
            else:
                # 類似度制御が無効な場合：各ユニットに独立したテンプレートを生成
                for unit in gt_units.units:
                    if not is_valid_template(unit):
                        unit.spikeTemp = BaseTemplate.generate(settings)

        elif settings["spikeSettings"]["spikeType"] == "template":
            # ファイルからテンプレートを読み込む
            template_file = Path(dir) / settings["spikeSettings"]["template"]["pathSpikeList"]
            if not template_file.exists():
                logging.error(f"テンプレートファイルが見つかりません: {template_file}")
                return False
            
            templates = BaseTemplate.load_spike_templates(template_file)
            for i, unit in enumerate(gt_units.units):
                unit.spikeTemp = templates[i] if i < len(templates) else templates[-1]
                if i >= len(templates):
                    logging.warning(f"テンプレート不足: ユニット{i}には最後のテンプレートを使用")
        else:
            raise ValueError(f"Invalid spike type: {settings['spikeSettings']['spikeType']}")

        # 信号生成
        logging.info(f"=== 信号生成 ===")
        
        # 全ユニットのテンプレートを検証
        for i, unit in enumerate(gt_units.units):
            if not is_valid_template(unit):
                logging.error(f"ユニット{i} (ID: {unit.id})のテンプレートが無効です")
                raise ValueError(f"ユニット{i}のテンプレートが無効です")
        
        logging.info(f"全{len(gt_units.units)}ユニットのテンプレートを確認 - 正常")
        
        # ノイズ信号の生成
        duration_samples = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
        drift = DriftNoise.generate(settings) if settings["driftSettings"]["enable"] else np.zeros(duration_samples)
        powerLineNoise = PowerLineNoise.generate(settings) if settings["powerNoiseSettings"]["enable"] else np.zeros(duration_samples)

        for contact in tqdm(probe.contacts, total=len(probe.contacts)):
            spike_signal = np.zeros(duration_samples)
            
            for unit in gt_units.units:
                scaled_amps = calculate_scaled_spike_amplitude(
                    unit.spikeAmpList, 
                    calculate_distance_two_objects(unit, contact), 
                    settings["spikeSettings"]["attenTime"]
                )
                spike_signal = addSpikeToSignal(
                    spike_signal, 
                    unit.spikeTimeList, 
                    unit.spikeTemp.template, 
                    scaled_amps
                )
            
            contact.set_signal("spike", spike_signal)
            contact.set_signal("drift", drift)
            contact.set_signal("power", powerLineNoise)
            

        # データの保存
        logging.info(f"=== データの保存 ===")
        if saveDir is None:
            logging.error("保存先ディレクトリがNoneです")
            return False
        
        noise_units_list = bg_units.units if (settings["noiseSettings"]["noiseType"] == "model" and bg_units) else None
        save_data(saveDir, gt_units.units, probe.contacts, noise_units=noise_units_list, fs=settings["baseSettings"]["fs"])
        logging.info(f"データ保存完了: {saveDir}")

        # プロット表示（オプション）
        if plot:
            plot_main(gt_units.units, noise_units_list, probe.contacts, str(dir.name), saveDir)

        logging.info(f"=== 実験完了 ===")
        return CarsObject(settings, gt_units.units, probe.contacts, noise_units_list)
        
    except Exception as e:
        logging.error(f"実験でエラー発生 - {e}", exc_info=True)
        return False

def init_logging(debug: bool):
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def init_example_dir(example_dir: Path):
    try:
        example_dir.exists()
    except Exception as e:
        logging.error(f"実験ディレクトリが見つかりません: {example_dir}", exc_info=True)
        raise e
    return example_dir

def find_conditions(dir: Path, pattern: str):
    founds = list(dir.glob(f"{pattern}"))
    try:
        for f in founds:
            f.exists()
    except Exception as e:
        logging.error(f"条件ファイルが見つかりません: {dir}/{pattern}", exc_info=True)
        raise e
    logging.info(f"該当した条件数: {len(founds)}")
    for i, condition_name in enumerate(founds, 1):
        logging.info(f"{i}. {condition_name}")
    return [f.stem for f in founds]
    

def main(project_root: Path, args: argparse.Namespace):
    """メイン関数 - 複数の条件を実行"""
    try:
        init_logging(args.debug)
        # ベースディレクトリからの相対パスとして扱う
        if args.test:
            example_dir = project_root / TEST_DIR / "example"
            init_example_dir(example_dir)
            logging.info(f"実験ディレクトリ: {example_dir}")
            # 条件ファイルの検索
            condition_names = find_conditions(example_dir, args.conditions)
            # 各条件を順次実行
            success_count = 0
            for i, condition_name in enumerate(condition_names, 1):
                logging.info(f"\n[{i}/{len(condition_names)}] {condition_name} を実行中...")
                exam = example_dir/condition_name
                if run(exam, plot=args.plot, settings=Path(exam)/SETTINGS_FILE, units=Path(exam)/UNITS_FILE, probe=Path(exam)/PROBE_FILE):
                    success_count += 1
                else:
                    logging.error(f"実験が失敗しました: {condition_name}")
            
            logging.info(f"\n=== 実行完了 ===")
            logging.info(f"成功: {success_count}/{len(condition_names)}")
        else:
            example_dir = Path(args.example_dir)
            init_example_dir(example_dir)
            logging.info(f"実験ディレクトリ: {example_dir}")
            # 条件ファイルの検索
            condition_names = find_conditions(example_dir, args.conditions)
        
            # 各条件を順次実行
            success_count = 0
            for i, condition_name in enumerate(condition_names, 1):
                logging.info(f"\n[{i}/{len(condition_names)}] {condition_name} を実行中...")
                exam = example_dir/condition_name
                if run(exam, settings=Path(exam)/SETTINGS_FILE, units=Path(exam)/UNITS_FILE, probe=Path(exam)/PROBE_FILE, plot=args.plot):
                    success_count += 1
                else:
                    logging.error(f"実験が失敗しました: {condition_name}")
            
            logging.info(f"\n=== 実行完了 ===")
            logging.info(f"成功: {success_count}/{len(condition_names)}")
        
    except Exception as e:
        logging.error(f"メイン処理でエラー発生: {e}", exc_info=True)

