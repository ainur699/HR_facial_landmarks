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
import sys
sys.argv.append('--cfg')
sys.argv.append('experiments/300w/face_alignment_300w_hrnet_w18_precise.yaml')
sys.argv.append('--model-file')
sys.argv.append('checkpoints/model_best.pth')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
import numpy as np
from PIL import Image
import cv2
import datetime

import glob
from tqdm import tqdm

from lib.core.evaluation import decode_preds
from lib.utils.transforms import crop

from lib.datasets.voxtrain_dataset import VoxCelebDataset
from torch.utils.data import DataLoader

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

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = list(config.GPUS)
    model = models.get_face_alignment_net(config)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    #load model
    state_dict = torch.load(args.model_file).state_dict()
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    model.eval()
    
    #dataset
    root_dir = 'D:/Github/video-preprocessing/vox1-png'
    dataset = VoxCelebDataset(root_dir, is_train=True)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            paths = data['path']
            imgs = data['image'].cuda()
            trf_inv = data['trf_inv']

            output = model(imgs)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, trf_inv)

            preds = preds.numpy()
            for pts, path in zip(preds, paths):
                save_path = os.path.join(root_dir, 'train_pt', path.replace('.png', '.txt'))
                dir, _ = os.path.split(save_path)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                f = open(save_path, 'w')
                for pt in pts:
                    f.write(str(pt[0]) + ' ')
                    f.write(str(pt[1]) + ' ')
                f.close()

                ###########
                bbox = cv2.boundingRect(pts)

                l = int(max(bbox[0] - 0.5 * bbox[2] / 2.0, 0))
                t = int(max(bbox[1] - 0.5 * bbox[3] / 2.0, 0))
                r = int(bbox[0] + bbox[2] + 0.5 * bbox[2] / 2.0)
                b = int(bbox[1] + bbox[3] + 0.5 * bbox[3] / 2.0)

                source = cv2.imread(os.path.join(root_dir, 'train', path))
                source = source[t:b, l:r, ...]
                ratio = 1024.0 / source.shape[1]
                source = cv2.resize(source, None, fx=ratio, fy=ratio)
                for k in pts:
                    source = cv2.circle(source, (int(ratio*(k[0] - l)), int(ratio*(k[1] - t))), 3, (0,255,0),-1,lineType=8)
                cv2.imwrite('d:/test/' + os.path.basename(path), source)
                #############


if __name__ == '__main__':
    main()

