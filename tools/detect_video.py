# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.argv.append('--cfg')
sys.argv.append('experiments/300w/face_alignment_300w_hrnet_w18_precise.yaml')
sys.argv.append('--model-file')
sys.argv.append('checkpoints/model_best.pth')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import json

import lib.models as models
from lib.config import config, update_config
from lib.core.evaluation import decode_preds
from lib.datasets.voxtrain_dataset import VideoDataset


def test(video_path):
    landmarks_name = video_path.replace('train', 'landmarks').replace('.mp4', '.json')
    with open(landmarks_name) as f:
        landmarks = json.load(f)

    cap = cv2.VideoCapture(video_path)

    i = 0
    ret, frame = cap.read()

    writer = cv2.VideoWriter(f'd:/test/{os.path.basename(video_path)}', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), frame.shape[:2])

    while ret:
        if str(i) in landmarks:
            pts = landmarks[str(i)]
            for pt in pts:
                frame = cv2.circle(frame, tuple(map(int, pt)), 2, (0,255,0), -1)
        i += 1
            
        writer.write(frame)
        ret, frame = cap.read()


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
    root_dir = 'D:/Github/video-preprocessing/vox-video/train'
    videos = glob(os.path.join(root_dir, '*.mp4'))
    for video_name in tqdm(videos):
        dataset = VideoDataset(video_name)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        landmarks = {}

        with torch.no_grad():
            for data in dataloader:
                frame = data['frame'].cuda()
                trf_inv = data['trf_inv']
                idx = data['id'].numpy()

                output = model(frame)
                score_map = output.data.cpu()
                preds = decode_preds(score_map, trf_inv)

                preds = preds.numpy()
                for pts, i in zip(preds, idx):
                    landmarks[str(i)] = pts.tolist()

        save_path = video_name.replace('train', 'landmarks').replace('.mp4', '.json')
        with open(save_path, 'w') as f:
            json.dump(landmarks, f)

        ##
        #test(video_name)
        ##


if __name__ == '__main__':
    main()

