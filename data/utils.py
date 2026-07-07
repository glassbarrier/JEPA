# TODO: 基于 le-wm-test/utils.py 提取
# 提取：只保留get_img_preprocessor和get_column_normalizer

import torch

def get_img_preprocessor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    返回图像预处理标准化参数
    """
    return {'mean': mean, 'std': std}

def get_column_normalizer():
    """
    占位符：如果后续需要处理表格数据，可在此扩展
    """
    return None