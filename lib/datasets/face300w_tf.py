import os
import random

import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import cv2

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def transform_data(x, y, data_root, is_train, scale_factor, rot_factor, flip, sigma, label_type, input_size, output_size):
    image_path = os.path.join(data_root.encode(), x)
    image_path = os.path.normpath(image_path) 
    tl = y[0:2]
    tr = y[2:4]
    br = y[4:6]
    pts = y[6:]
    pts = pts.reshape(-1, 2)

    #r = 0
    #scale = 1
    #if is_train:
    #    scale = scale * (random.uniform(1 - scale_factor, 1 + scale_factor)) if random.random() <= 0.6 else 0
    #    r = random.uniform(-rot_factor, rot_factor) if random.random() <= 0.6 else 0
    #    if random.random() <= 0.5 and flip:
    #        img = np.fliplr(img)
    #        pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
    #        center[0] = img.shape[1] - center[0]

    pt_src = np.float32([tl, tr, br])
    pt_target = np.float32([[0,0], [output_size[0] - 1, 0], [output_size[0] - 1, output_size[1] - 1]])
    pt_crop = np.float32([[0,0], [input_size[0] - 1, 0], [input_size[0] - 1, input_size[1] - 1]])

    trf_img = cv2.getAffineTransform(pt_src, pt_crop)
    trf_pt = cv2.getAffineTransform(pt_src, pt_target)
    trf_pt_inv = cv2.getAffineTransform(pt_target, pt_src)
    trf_pt_inv.resize((3,3))
    trf_pt_inv[2][2] = 1

    target = np.zeros((pts.shape[0], output_size[0], output_size[1]), dtype=np.float32)
    tpts = np.zeros((pts.shape[0], 2), dtype=np.float32)

    for i in range(pts.shape[0]):
        tpts[i] = cv2.transform(np.array([[pts[i]]]), trf_pt)[0][0]
        target[i] = generate_target(target[i], tpts[i], sigma, label_type=label_type)

    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
    img = cv2.warpAffine(img, trf_img, tuple(input_size))
    img = (img/255.0 - mean) / std
    img = img.transpose([2, 0, 1])

    return img, target, np.float32(pts), np.float32(tpts), np.float32(trf_pt_inv), image_path


def get_300W_dataset(cfg, is_train=True):
    tf.config.set_visible_devices([], 'GPU')

    # specify annotation file for dataset
    if is_train:
        csv_file = cfg.DATASET.TRAINSET
    else:
        csv_file = cfg.DATASET.TESTSET
    
    data_root    = cfg.DATASET.ROOT
    input_size   = cfg.MODEL.IMAGE_SIZE
    output_size  = cfg.MODEL.HEATMAP_SIZE
    sigma        = cfg.MODEL.SIGMA
    scale_factor = cfg.DATASET.SCALE_FACTOR
    rot_factor   = cfg.DATASET.ROT_FACTOR
    label_type   = cfg.MODEL.TARGET_TYPE
    flip         = cfg.DATASET.FLIP
    
    # load annotations
    raw_dataset = pd.read_csv(csv_file, header=None).values
    img_names = raw_dataset[:, 0]
    landmarks = raw_dataset[:, 1:].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((img_names, landmarks))

    numpy_func = lambda x, y: transform_data(x, y, data_root, is_train, scale_factor, rot_factor, flip, sigma, label_type, input_size, output_size)

    def tf_func(x, y):
        img, target, pts, tpts, trf_inv, image_path = tf.numpy_function(numpy_func, [x, y], (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.string))

        meta = {'pts': pts, 'tpts' : tpts, 'trf_inv': trf_inv, 'img_name': image_path}
        return img, target, meta

    dataset = dataset.map(tf_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #test
    #for i, (inp, target, meta) in enumerate(dataset.as_numpy_iterator()):
    #    img = inp
    #    img = img.transpose([1, 2, 0])
    #    img = 255 * (img * std + mean)
    #    tpts = meta['tpts']
    #
    #    for pt in tpts:
    #        img = cv2.circle(img, (int(4 * pt[0]), int(4 * pt[1])), 1, (0,255,0),-1,lineType=8)
    #    cv2.imwrite('label.png', img)

    return dataset


