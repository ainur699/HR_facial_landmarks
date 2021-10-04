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
sys.argv.append('output/300W/face_alignment_300w_hrnet_w18_precise/model_best.pth')
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

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


import tensorflow as tf

def main():
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    args = parse_args()

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

    model.eval()

    #image_dir = 'D:/Datasets/IBUG/Test/01_Indoor/'
    #bbox_dir = 'D:/results/yolo_result/yolo_tiny_predictions/'
    
    #image_dir = 'D:/PhotolabImages/good-data/'
    #bbox_dir = 'D:/PhotolabImages/good-data_ellipses/txt/'
    #dst_dir = 'D:/PhotolabImages/good-data_hr_points/txt/'

    #image_dir = 'D:/Images/'
    #bbox_dir = 'D:/Images/bbox/'
    dst_dir = 'D:/test/'

    #filenames = glob.glob('D:\\Datasets\\GenFaces\\regular\\*\\*.png')
    #filenames = glob.glob('D:/test_dataset/*.jpg')
    filenames = glob.glob('D:/photolab_test/imgs/*')

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    time_all = 0
    img_num = 0    

    with torch.no_grad():
        #for filename in os.listdir(image_dir):
        for filename in tqdm(filenames):
            #if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                #image_path = os.path.join(image_dir, filename)
                #image_path = os.path.normpath(image_path)
                image_path = filename

                #bbox_path = os.path.join(bbox_dir, filename + '.txt')
                #bbox_path = os.path.normpath(bbox_path)
                #bbox_path = filename.replace('.png', '_bbox.txt')
                bbox_path = filename.replace('imgs', 'bbox').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')


                f = open(bbox_path, 'r')
                bbox_pts = f.readline().split(',')

                if len(bbox_pts) != 6:
                    continue
                bbox_pts = [float(v) for v in bbox_pts]

                tl, tr, br = bbox_pts[0:2], bbox_pts[2:4], bbox_pts[4:6]
                src = np.float32([tl, tr, br])
                pt_input = np.float32([[0,0], [511, 0], [511, 511]])
                pt_target = np.float32([[0,0], [127, 0], [127, 127]])

                trf = cv2.getAffineTransform(src, pt_input)
                trf_inv = cv2.getAffineTransform(pt_target, src).astype(np.float32)

                source = cv2.imread(image_path)
                #source = tf.io.read_file(image_path)
                #source = tf.image.decode_image(source, channels=3).numpy()
                #source = source.astype(np.float32)
                

                img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
                #img = source.copy()

                img = cv2.warpAffine(img, trf, (512, 512))
                img = (img/255.0 - mean) / std
                img = img.transpose([2, 0, 1])
                img = torch.Tensor(img)
                img = img[None, :]

                #start = datetime.datetime.now()
                output = model(img)
                #stop = datetime.datetime.now()
                #print((stop - start).microseconds / 1000.0, 'ms...')

                #time_all = time_all + (stop - start).microseconds / 1000.0
                #img_num = img_num + 1

                score_map = output.data.cpu()
                preds = decode_preds(score_map, [trf_inv])

                # output
                pts = preds[0].numpy()
#                bbox = cv2.boundingRect(pts)
#
#                l = int(max(bbox[0] - 0.5 * bbox[2] / 2.0, 0))
#                t = int(max(bbox[1] - 0.5 * bbox[3] / 2.0, 0))
#                r = int(bbox[0] + bbox[2] + 0.5 * bbox[2] / 2.0)
#                b = int(bbox[1] + bbox[3] + 0.5 * bbox[3] / 2.0)
#
#                source = source[t:b, l:r, ...]
#                ratio = 1024.0 / source.shape[1]
#                source = cv2.resize(source, None, fx=ratio, fy=ratio)
#                source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
#                for k in pts:
#                    source = cv2.circle(source, (int(ratio*(k[0] - l)), int(ratio*(k[1] - t))), 3, (0,255,0),-1,lineType=8)
#                cv2.imwrite(dst_dir + os.path.basename(filename), source)

                f = open(bbox_path.replace('bbox', 'landmarks'), 'w')
                for pt in pts:
                    f.write(str(pt[0]) + ' ')
                    f.write(str(pt[1]) + ' ')
                f.close()

    #print('mean time', time_all / img_num)

if __name__ == '__main__':
    main()

