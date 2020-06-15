# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.argv.append('--cfg')
sys.argv.append('experiments/300w/face_alignment_300w_hrnet_w18_precise.yaml')
sys.argv.append('--model-file')
#sys.argv.append('hrnetv2_pretrained/HR18-300W.pth')
sys.argv.append('output/300W/face_alignment_300w_hrnet_w18_precise/model_best.pth')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
from lib.datasets.face300w_tf import get_300W_dataset


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'test')

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file).state_dict()
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)


    #dataset_type = get_dataset(config)
    #
    #test_loader = DataLoader(
    #    dataset=dataset_type(config,
    #                         is_train=False),
    #    batch_size=1,#config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    #    shuffle=False,
    #    num_workers=0,#config.WORKERS,
    #    pin_memory=config.PIN_MEMORY
    #)
    train_loader = get_300W_dataset(config, is_train=False)

    nme = function.inference(config, train_loader, model)


if __name__ == '__main__':
    main()

