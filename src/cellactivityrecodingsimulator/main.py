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
from .carsIO import save_data
from .Noise import RandomNoise, DriftNoise, PowerLineNoise
from .Template import BaseTemplate
from .tools import make_save_dir
from .plot import plot_GTUnits, plot_Signals
from .ProbeObject import ProbeObject
# ベースディレクトリを固定
BASE_DIR = Path("simulations")
TEST_DIR = Path("test")
SETTINGS_FILE = Path("settings.json")
UNITS_FILE = Path("cells.json")
PROBE_FILE = Path("probe.json")

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def init_run(settings: dict, dir: Path, verbose):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # 保存ディレクトリの作成
    if settings["baseSettings"]["pathSaveDir"] is None:
        pathSaveDir = dir
    else:
        pathSaveDir = settings["baseSettings"]["pathSaveDir"]
        
    saveDir = make_save_dir(pathSaveDir)
    settings["baseSettings"]["pathSaveDir"] = saveDir

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
        logging.info(f"=== 実験開始 ===")
        settings = Settings.load(settings).to_dict()
        logging.info(f"設定ファイル読み込み完了")
        
        # セルデータの読み込み
        gt_units = GTUnitsObject.load(units, settings=settings)
        logging.info(f"セルデータ読み込み完了: {gt_units.get_units_num()} units")
        
        # サイトデータの読み込み
        probe = ProbeObject.load(probe)
        logging.info(f"サイトデータ読み込み完了: {probe.get_contacts_num()} contacts")

        init_run(settings, Path(dir), verbose) 

        # ノイズの適用
        logging.info(f"=== ノイズ生成と記録点への適用 ===")
        if settings["noiseSettings"]["noiseType"] == "truth":
            logging.warning("truth noise type is not implemented")
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                noise = RandomNoise.generate("normal", settings)
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
            bg_units = BGUnitsObject.generate(settings, probe)
            for contact in tqdm(probe.contacts, desc="ノイズ割振中", total=probe.get_contacts_num()):
                bg_units.make_background_activity(contact, settings["spikeSettings"]["attenTime"], settings)

        elif settings["noiseSettings"]["noiseType"] == "none":
            duration_samples = int(settings["baseSettings"]["duration"] * settings["baseSettings"]["fs"])
            for contact in probe.contacts:
                contact.set_signal("background", np.zeros(duration_samples))

        else:
            noiseType = settings["noiseSettings"]["noiseType"]
            raise ValueError(f"Invalid noise type: {noiseType}")
        
        # スパイクテンプレートの生成
        logging.info(f"=== スパイク活動の生成 ===")
        
        def is_valid_template(unit):
            """ユニットのテンプレートが有効かどうかをチェック"""
            return (
                hasattr(unit, 'templateObject') and 
                unit.get_templateObject() is not None and
                hasattr(unit.get_templateObject(), 'template') and
                len(unit.get_templateObject().get_template()) > 1
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
                                unit.set_templateObject(settings=settings)
                            base_template = unit.get_templateObject()
                        else:
                            new_template = base_template.generate_similar_template(settings)
                            unit.set_templateObject(template=new_template)
            else:
                # 類似度制御が無効な場合：各ユニットに独立したテンプレートを生成
                for unit in gt_units.units:
                    if not is_valid_template(unit):
                        unit.set_templateObject(settings=settings)

        elif settings["spikeSettings"]["spikeType"] == "template":
            logging.warning("template spike type is not implemented")
            # ファイルからテンプレートを読み込む
            template_file = Path(dir) / settings["spikeSettings"]["template"]["pathSpikeList"]
            if not template_file.exists():
                logging.error(f"テンプレートファイルが見つかりません: {template_file}")
                return False
            
            templates = BaseTemplate.load_spike_templates(template_file)
            for i, unit in enumerate(gt_units.units):
                unit.templateObject = templates[i] if i < len(templates) else templates[-1]
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
        drift = DriftNoise.generate(settings)
        powerLineNoise = PowerLineNoise.generate(settings)

        for contact in tqdm(probe.contacts, total=len(probe.contacts)):
            for unit in gt_units.units:
                contact.add_spikes(unit, settings)
            
            contact.set_signal("drift", drift.signal)
            contact.set_signal("power", powerLineNoise.signal)

        # データの保存
        logging.info(f"=== データの保存 ===")
        if settings["baseSettings"]["pathSaveDir"] is None:
            logging.error("保存先ディレクトリがNoneです")
            return False
        
        noise_units_list = bg_units.units if (settings["noiseSettings"]["noiseType"] == "model" and bg_units) else None
        save_data(settings["baseSettings"]["pathSaveDir"], gt_units.units, probe.contacts, noise_units=noise_units_list, fs=settings["baseSettings"]["fs"])
        logging.info(f"データ保存完了")

        # プロット表示（オプション）
        if plot:
            plot_GTUnits(gt_units.units, with_probe=True, probe=probe)
            plot_Signals(probe.contacts)

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

