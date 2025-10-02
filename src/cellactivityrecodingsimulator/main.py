import numpy as np
import random
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
import sys
from probeinterface import Probe

from .CarsObject import CarsObject
from .Settings import Settings
from .Settings import default_settings, default_cells, default_sites
from .carsIO import load_settings_file, load_cells_from_json, load_sites_from_Probe, load_sites_from_json, save_data, load_noise_file, load_spike_templates
from .Noise import RandomNoise, DriftNoise, PowerLineNoise
from .Template import make_similar_templates, GaborTemplate, ExponentialTemplate
from .generate import make_noise_cells, make_background_activity, make_spike_times
from .calculate import calculate_spike_max_amplitude, calculate_scaled_spike_amplitude, calculate_distance_two_objects
from .tools import addSpikeToSignal, make_save_dir
from .plot.main import plot_main
# ベースディレクトリを固定
BASE_DIR = Path("simulations")
TEST_DIR = Path("test")
SETTINGS_FILE = Path("settings.json")
CELLS_FILE = Path("cells.json")
PROBE_FILE = Path("probe.json")

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def init_run(settings: Settings, dir: Path):
    baseSettings = settings.rootSettings.to_dict()
    validate_settings(settings)
    set_seed(baseSettings)
    # 保存ディレクトリの作成
    if baseSettings["pathSaveDir"] is None:
        pathSaveDir = dir
    else:
        pathSaveDir = baseSettings["pathSaveDir"]
        
    saveDir = make_save_dir(pathSaveDir)
    return saveDir

def set_seed(baseSettings: dict):
    random.seed(baseSettings["random_seed"])
    np.random.seed(baseSettings["random_seed"])

def validate_settings(settings: Settings):
    logging.info("設定の検証を開始")
    errors = settings.validate()
    if len(errors) == 0:
        logging.info("設定の有効性を確認")
    else:
        logging.error(f"無効な設定を確認. 処理を停止: {errors}")
        sys.exit(1)

def load_files(object: Path|Probe, type: str):
    if type == "settings":
        return load_settings_file(object)
    elif type == "cells":
        return load_cells_from_json(object)
    elif type == "probe" and isinstance(object, Path):
        return load_sites_from_json(object)
    elif type == "probe" and isinstance(object, Probe):
        return load_sites_from_Probe(object)
    else:
        raise ValueError(f"Invalid file type")

def run(dir: Path, settings:Path=None, cells:Path=None, probe:Path|Probe=None, plot: bool=False):
    """単一の実験を実行する"""
    try:
        logging.info(f"=== 実験開始 ===")
        
        if settings is not None:
            settings = load_files(settings, "settings")
        else:
            settings = Settings(default_settings())
        logging.info(f"設定ファイル読み込み完了")
        
        # セルデータの読み込み
        if cells is not None:
            cells = load_files(cells, "cells")
        else:
            cells = default_cells()
        logging.info(f"セルデータ読み込み完了: {len(cells)} cells")
        
        # サイトデータの読み込み
        if probe is not None:
            sites = load_files(probe, "probe")
        else:
            sites = default_sites()
        logging.info(f"サイトデータ読み込み完了: {len(sites)} sites")

        saveDir = init_run(settings, Path(dir)) 

        baseSettings = settings.rootSettings.to_dict()
        spikeSettings = settings.spikeSetting.to_dict()
        noiseSettings = settings.noiseSettings.to_dict()
        driftSettings = settings.driftSettings.to_dict()
        powerNoiseSettings = settings.powerNoiseSettings.to_dict()
        templateSimilarityControlSettings = settings.templateSimilarityControlSettings.to_dict()

        # ノイズの適用
        if noiseSettings["noiseType"] == "truth":
            truthNoiseSettings = noiseSettings["truth"]
            for site in tqdm(sites, total=len(sites)):
                site.set_signal(
                    "background", 
                    load_noise_file(
                        truthNoiseSettings["pathNoise"]
                        ))
        elif noiseSettings["noiseType"] == "normal":  
            normalNoiseSettings = noiseSettings["normal"]
            for site in tqdm(sites, total=len(sites)):
                site.set_signal("background", 
                RandomNoise(
                    baseSettings["fs"], 
                    baseSettings["duration"], 
                    normalNoiseSettings["amplitude"]
                    ).generate("normal"))
        elif noiseSettings["noiseType"] == "gaussian":
            gaussianNoiseSettings = noiseSettings["gaussian"]
            for site in tqdm(sites, total=len(sites)):
                site.set_signal("background", 
                RandomNoise(
                    baseSettings["fs"], 
                    baseSettings["duration"], 
                    gaussianNoiseSettings["amplitude"], 
                    gaussianNoiseSettings["location"], 
                    gaussianNoiseSettings["scale"]
                    ).generate("gaussian"))
        elif noiseSettings["noiseType"] == "model":
            modelNoiseSettings = noiseSettings["model"]
            # ノイズ細胞を生成してサイトに追加
            noise_cells = make_noise_cells(
                baseSettings["duration"], 
                baseSettings["fs"], 
                sites, 
                spikeSettings,
                noiseSettings
                )
            for site in tqdm(sites, total=len(sites)):
                site.set_signal("background", 
                make_background_activity(
                    baseSettings["duration"], 
                    baseSettings["fs"], 
                    noise_cells, 
                    site, 
                    spikeSettings["attenTime"]
                    ))
        elif noiseSettings["noiseType"] == "none":
            for site in tqdm(sites, total=len(sites)):
                site.set_signal("background", 
                np.zeros(int(baseSettings["duration"] * baseSettings["fs"])))
        else:
            raise ValueError(f"Invalid noise type")
        
        # スパイクテンプレートの読み込み
        if spikeSettings["spikeType"] in ["gabor", "exponential"]:
            if templateSimilarityControlSettings["enable"]:
                group_ids = list(set([cell.group for cell in cells]))
                for group_id in group_ids:
                    group_cells = [cell for cell in cells if cell.group == group_id]
                    spikeTemplates = make_similar_templates(
                        baseSettings["fs"], 
                        templateSimilarityControlSettings,
                        spikeSettings,
                        len(group_cells)
                        )
                    for i, cell in enumerate(group_cells):
                        cell.spikeTemp = spikeTemplates[i]
                        cell.spikeTimeList = make_spike_times(
                            baseSettings["duration"], 
                            baseSettings["fs"], 
                            spikeSettings["rate"], 
                            spikeSettings["refractoryPeriod"]
                            )
                        logging.debug(f"cell{cell.id}.spikeTimeList: {len(cell.spikeTimeList)}")
                        for t in cell.spikeTimeList:
                            cell.spikeAmpList.append(
                                calculate_spike_max_amplitude(
                                    spikeSettings["amplitudeMax"], 
                                    spikeSettings["amplitudeMin"])
                                )
            else:
                for i, cell in enumerate(cells):
                    if spikeSettings["spikeType"] == "gabor":
                        cell.spikeTemp = GaborTemplate(baseSettings["fs"], spikeSettings).generate()
                    elif spikeSettings["spikeType"] == "exponential":
                        cell.spikeTemp = ExponentialTemplate(baseSettings["fs"], spikeSettings).generate()
                    else:
                        raise ValueError(f"Invalid spike type")
                    cell.spikeTimeList = make_spike_times(
                        baseSettings["duration"], 
                        baseSettings["fs"], 
                        spikeSettings["rate"], 
                        spikeSettings["refractoryPeriod"]
                        )
                    cell.spikeAmpList = [calculate_spike_max_amplitude(
                        spikeSettings["amplitudeMax"], 
                        spikeSettings["amplitudeMin"]
                        ) for _ in range(len(cell.spikeTimeList))]

        elif spikeSettings["spikeType"] == "template":
            templateSettings = spikeSettings["template"]
            template_file = Path(dir) / templateSettings["pathSpikeList"]
            if not template_file.exists():
                logging.error(f"テンプレートファイルが見つかりません: {template_file}")
                return False
            spikeTemplates = load_spike_templates(template_file)
            for i, cell in enumerate(cells):
                cell.spikeTemp = spikeTemplates[i]
                cell.spikeTimeList = make_spike_times(
                    baseSettings["duration"], 
                    baseSettings["fs"], 
                    spikeSettings["rate"], 
                    spikeSettings["refractoryPeriod"])
                cell.spikeAmpList = [calculate_spike_max_amplitude(
                    spikeSettings["amplitudeMax"], 
                    spikeSettings["amplitudeMin"]
                    ) for _ in range(len(cell.spikeTimeList))]
        else:
            raise ValueError(f"Invalid spike type")

        # 信号生成
        logging.info("信号生成開始...")
        if driftSettings["enable"]:
            drift = DriftNoise(
                baseSettings["fs"], 
                baseSettings["duration"], 
                driftSettings
                ).generate()
        else:
            drift = np.zeros(int(baseSettings["duration"] * baseSettings["fs"]))
        if powerNoiseSettings["enable"]:
            powerLineNoise = PowerLineNoise(
                baseSettings["fs"], 
                baseSettings["duration"], 
                powerNoiseSettings
                ).generate()
        else:
            powerLineNoise = np.zeros(int(baseSettings["duration"] * baseSettings["fs"]))

        for site in tqdm(sites, total=len(sites)):
            spike = np.zeros(int(baseSettings["duration"] * baseSettings["fs"]))
            for cell in cells:
                scaledSpikeAmpList = calculate_scaled_spike_amplitude(
                    cell.spikeAmpList, 
                    calculate_distance_two_objects(cell, site), 
                    spikeSettings["attenTime"]
                    )
                spike = addSpikeToSignal(
                    spike, 
                    cell.spikeTimeList, 
                    cell.spikeTemp, 
                    scaledSpikeAmpList
                    )
            site.set_signal("spike", spike)
            site.set_signal("drift", drift)
            site.set_signal("power", powerLineNoise)
            

        # データの保存
        logging.info(f"保存先ディレクトリ: {saveDir}")
        logging.info(f"saveDirの型: {type(saveDir)}")
        logging.info(f"saveDirの絶対パス: {saveDir.absolute() if hasattr(saveDir, 'absolute') else 'N/A'}")
        
        if saveDir is None:
            logging.error("保存先ディレクトリがNoneです。データ保存をスキップします。")
            return False
            
        save_data(
            saveDir, 
            cells, 
            sites, 
            noise_cells=noise_cells if noiseSettings["noiseType"] == "model" else None, 
            fs=baseSettings["fs"])
        logging.info(f"データ保存完了: {saveDir}")

        # プロット表示（オプション）
        if plot:
            plot_main(cells, noise_cells, sites, dir)

        logging.info(f"=== 実験完了 ===")
        return CarsObject(settings, cells, sites, noise_cells if noiseSettings["noiseType"] == "model" else None)
        
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
                if run(exam, plot=args.plot):
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
                if run(exam, settings=exam/SETTINGS_FILE, cells=exam/CELLS_FILE, probe=exam/PROBE_FILE, plot=args.plot):
                    success_count += 1
                else:
                    logging.error(f"実験が失敗しました: {condition_name}")
            
            logging.info(f"\n=== 実行完了 ===")
            logging.info(f"成功: {success_count}/{len(condition_names)}")
        
    except Exception as e:
        logging.error(f"メイン処理でエラー発生: {e}", exc_info=True)

