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
    scale = y[0]
    center = [y[1], y[2]]
    pts = y[3:]
    
    pts = pts.reshape(-1, 2)
    scale *= 1.25
    nparts = pts.shape[0]
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # debug
    #for pt in pts:
    #   img = cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0,255,0),-1,lineType=8)
    #cv2.imwrite('label_init.png', img)

    r = 0
    if is_train:
        scale = scale * (random.uniform(1 - scale_factor, 1 + scale_factor))
        r = random.uniform(-rot_factor, rot_factor) if random.random() <= 0.6 else 0
        if random.random() <= 0.5 and flip:
            img = np.fliplr(img)
            pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
            center[0] = img.shape[1] - center[0]

    img = crop(img, center, scale, input_size, rot=r)

    target = np.zeros((nparts, output_size[0], output_size[1]), dtype=np.float32)
    tpts = pts.copy()

    for i in range(nparts):
        if tpts[i, 1] > 0:
            tpts[i, 0:2] = transform_pixel(tpts[i, 0:2], center, scale, output_size, rot=r)
            target[i] = generate_target(target[i], tpts[i], sigma, label_type=label_type)

    # debug
    #for pt in tpts:
    #    img = cv2.circle(img, (int(4 * pt[0]), int(4 * pt[1])), 1, (0,255,0),-1,lineType=8)
    #cv2.imwrite('label.png', img)

    img = (img/255.0 - mean) / std
    img = img.transpose([2, 0, 1])

    center = np.array(center, dtype=np.float32)
    scale  = np.array(scale, dtype=np.float32)
    pts    = np.array(pts, dtype=np.float32)
    tpts   = np.array(tpts, dtype=np.float32)

    return img, target, center, scale, pts, tpts, image_path


def get_300W_dataset(cfg, is_train=True):
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
    raw_dataset = pd.read_csv(csv_file).values
    img_names = raw_dataset[:, 0]
    landmarks = raw_dataset[:, 1:].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((img_names, landmarks))

    numpy_func = lambda x, y: transform_data(x, y, data_root, is_train, scale_factor, rot_factor, flip, sigma, label_type, input_size, output_size)

    def tf_func(x, y):
        img, target, center, scale, pts, tpts, image_path = tf.numpy_function(numpy_func, [x, y], (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.string))

        meta = {'center': center, 'scale': scale, 'pts': pts, 'tpts': tpts, 'img_name': image_path}
        return img, target, meta

    dataset = dataset.map(tf_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #test
    #for i, (inp, target, meta) in enumerate(dataset.as_numpy_iterator()):
    #    tpts = meta['tpts']
    #    img = inp
    #    img = img.transpose([1, 2, 0])
    #    img = 255 * (img * std + mean)
    #
    #    for pt in tpts:
    #        img = cv2.circle(img, (int(4 * pt[0]), int(4 * pt[1])), 1, (0,255,0),-1,lineType=8)
    #    cv2.imwrite('label.png', img)

    return dataset


