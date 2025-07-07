
import os
import logging
import pandas as pd
import json
import numpy as np
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def load_dataset_config():
    config_path = Path(__file__).parent / 'data_config.json'
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('DATASET_PATHS', {})
DATASET_PATHS = load_dataset_config()
def load_dataset(dataset_type, dataset_num):
    if dataset_type not in DATASET_PATHS:
        logger.error(f"不支持的数据集类型: {dataset_type}")
        return None, None, None, None
    if not isinstance(dataset_num, int) or not (1 <= dataset_num <= 5):
        logger.error(f"数据集编号必须是1-5的整数，收到: {dataset_num}")
        return None, None, None, None
    config = DATASET_PATHS[dataset_type]
    base_dir = Path(config['base_dir']).resolve()
    gold_standard_dir = Path(config['gold_standard_dir']).resolve()
    file_prefix = config['file_prefix']
    gold_standard_prefix = config['gold_standard_prefix']
    dataset_name = str(dataset_num)
    if file_prefix:
        time_series_file = base_dir / f"{file_prefix}_{dataset_name}_timeseries.tsv"
        wildtype_file = base_dir / f"{file_prefix}_{dataset_name}_wildtype.tsv"
        multifactorial_file = base_dir / f"{file_prefix}_{dataset_name}_multifactorial.tsv"
    else:
        time_series_file = base_dir / "timeseries.tsv"
        wildtype_file = base_dir / "wildtype.tsv"
        multifactorial_file = base_dir / "multifactorial.tsv"
    if gold_standard_prefix:
        gold_standard_file = gold_standard_dir / f"{gold_standard_prefix}_{dataset_name}.tsv"
    else:
        gold_standard_file = gold_standard_dir / "gold_standard.tsv"
    if not time_series_file.exists():
        logger.error(f"时间序列数据文件不存在: {time_series_file}")
        return None, None, None, None
    time_series_data = pd.read_csv(time_series_file, sep='\t', encoding='utf-8')
    logger.info(f"时间序列数据形状: {time_series_data.shape}")
    if not gold_standard_file.exists():
        logger.error(f"金标准数据文件不存在: {gold_standard_file}")
        return None, None, None, None
    gold_standard_data = pd.read_csv(gold_standard_file, sep='\t', encoding='utf-8')
    logger.info(f"金标准数据形状: {gold_standard_data.shape}")
    if not wildtype_file.exists():
        logger.error(f"野生型数据文件不存在: {wildtype_file}")
        return None, None, None, None
    wildtype_data = pd.read_csv(wildtype_file, sep='\t', encoding='utf-8')
    logger.info(f"野生型数据形状: {wildtype_data.shape}")
    if not multifactorial_file.exists():
        logger.warning(f"多因素数据文件不存在: {multifactorial_file}，使用野生型数据代替")
        multifactorial_data = wildtype_data.copy()
    else:
        multifactorial_data = pd.read_csv(multifactorial_file, sep='\t', encoding='utf-8')
    logger.info(f"多因素数据形状: {multifactorial_data.shape}")
    return time_series_data, gold_standard_data, wildtype_data, multifactorial_data
def set_dataset_paths(dataset_type, dataset_num):
    if dataset_type not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    paths = DATASET_PATHS[dataset_type]
    base_dir = Path(paths['base_dir']).resolve()
    gold_standard_dir = Path(paths['gold_standard_dir']).resolve()
    file_prefix = paths['file_prefix']
    gold_standard_prefix = paths['gold_standard_prefix']
    expression_file = base_dir / f"{file_prefix}_{dataset_num}_timeseries.tsv"
    gold_standard_file = gold_standard_dir / f"{gold_standard_prefix}_{dataset_num}.tsv"
    return expression_file, gold_standard_file
def load_config(config_file="data_config.json"):
    global DATASET_PATHS
    try:
        config_path = Path(os.path.abspath(__file__)).parent / config_file
        if config_path.exists():
            with open(config_path, "r", encoding='utf-8') as f:
                config = json.load(f)
                if "DATASET_PATHS" in config:
                    DATASET_PATHS = config["DATASET_PATHS"]
                    logger.info(f"已从{config_path}加载数据集路径配置")
    except Exception as e:
        logger.warning(f"加载配置文件{config_path}失败: {str(e)}")
def save_config(config_file="data_config.json"):
    try:
        config_path = Path(os.path.abspath(__file__)).parent / config_file
        with open(config_path, "w", encoding='utf-8') as f:
            json.dump({"DATASET_PATHS": DATASET_PATHS}, f, indent=4, ensure_ascii=False)
        logger.info(f"数据集路径配置已保存到{config_path}")
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {str(e)}")
        return False
def set_dataset_path(dataset_type, base_dir=None, gold_standard_dir=None):
    if dataset_type not in DATASET_PATHS:
        DATASET_PATHS[dataset_type] = {
            'base_dir': "",
            'gold_standard_dir': "",
            'file_prefix': f"{dataset_type.lower()}_size100",
            'gold_standard_prefix': f"{dataset_type}_GoldStandard_Size100"
        }
    if base_dir:
        DATASET_PATHS[dataset_type]['base_dir'] = str(Path(base_dir).resolve())
    if gold_standard_dir:
        DATASET_PATHS[dataset_type]['gold_standard_dir'] = str(Path(gold_standard_dir).resolve())
    save_config()
    return True 
