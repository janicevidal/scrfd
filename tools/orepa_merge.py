import argparse
import os.path as osp

import numpy as np
import os
#import onnxruntime as rt
import torch

from mmdet.core import (build_model_from_cfg, generate_inputs_and_wrap_model,
                        preprocess_example_input)

import mmcv
from mmcv.runner import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--save', type=str, help='new checkpoint file path')
    args = parser.parse_args()
    return args


def build_model_from_cfg(config_path, checkpoint_path):
    """Build a model from config and load the given checkpoint.

    Args:
        config_path (str): the OpenMMLab config for the model we want to
            export to ONNX
        checkpoint_path (str): Path to the corresponding checkpoint

    Returns:
        torch.nn.Module: the built model
    """
    from mmdet.models import build_detector

    cfg = mmcv.Config.fromfile(config_path)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.cpu().eval()
    return model

if __name__ == '__main__':
    args = parse_args()
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    tmp_ckpt_file = None
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
        tmp_ckpt_file = args.checkpoint+"_slim.pth"
        torch.save(checkpoint, tmp_ckpt_file)
        print('remove optimizer params and save to', tmp_ckpt_file)
        checkpoint_path = tmp_ckpt_file
    
    model = build_model_from_cfg(args.config, checkpoint_path)
    
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
     
    torch.save(model.state_dict(), args.save)    