import os
import math
import numpy as np
import cv2
import glob
import torch
import torch.utils.data as data
import json


class VoxCelebDataset(data.Dataset):

    def __init__(self, root_dir, is_train):
        self.root_dir = root_dir

        if is_train:
            self.videos = glob.glob(os.path.join(self.root_dir, 'train/*/*.png'))
        else:
            self.videos = glob.glob(os.path.join(self.root_dir, 'test/*/*.png'))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.pt_input = np.float32([[0,0], [511, 0], [511, 511]])
        self.pt_target = np.float32([[0,0], [127, 0], [127, 127]])

    def load_img(self, image_path, affine):
        img = cv2.imread(image_path)

        if img is None:
            raise Exception('None Image')

        img = img[..., ::-1]
        img = cv2.warpAffine(img, affine, (512, 512))
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)

        return img

    def load_bbox(self, bbox_path):
        f = open(bbox_path, 'r')
        bbox_pts = f.readline().split(',')
        f.close()

        if len(bbox_pts) != 6:
            return None

        bbox_pts = [float(v) for v in bbox_pts]
        tl, tr, br = bbox_pts[0:2], bbox_pts[2:4], bbox_pts[4:6]
        bbox = np.float32([tl, tr, br])

        return bbox

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        bbox_path = self.videos[idx].replace('train', 'train_bbox').replace('.png', '.txt')
        bbox = self.load_bbox(bbox_path)
        if bbox is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        
        trf = cv2.getAffineTransform(bbox, self.pt_input)
        trf_inv = cv2.getAffineTransform(self.pt_target, bbox)

        image = self.load_img(self.videos[idx], trf)

        out = {}
        out['image'] = image
        out['trf_inv'] = trf_inv.astype(np.float32)
        out['path'] = os.path.join(*os.path.normpath(self.videos[idx]).split(os.sep)[-2:])
        return out


class VideoDataset(data.Dataset):

    def __init__(self, video_path):
        self.capture = cv2.VideoCapture(video_path)

        with open(video_path.replace('train','bbox_filtered').replace('.mp4', '.json')) as f:
            bboxes = json.load(f)

        self.bboxes = [ (k, v) for k, v in bboxes.items() if v ]

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.pt_input = np.float32([[0,0], [511, 0], [511, 511]])
        self.pt_target = np.float32([[0,0], [127, 0], [127, 127]])

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, i):
        idx = int(self.bboxes[i][0])

        bbox = self.bboxes[i][1]
        bbox = np.float32(list(bbox.values()))

        trf = cv2.getAffineTransform(bbox, self.pt_input)
        trf_inv = cv2.getAffineTransform(self.pt_target, bbox)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.capture.read()
        if not ret:
            raise Exception('None Image')
        
        frame = frame[..., ::-1]
        frame = cv2.warpAffine(frame, trf, (512, 512))
        frame = (frame/255 - self.mean) / self.std
        frame = frame.transpose([2, 0, 1])
        frame = frame.astype(np.float32)

        data = {}
        data['frame'] = frame
        data['id'] = idx
        data['trf_inv'] = trf_inv.astype(np.float32)

        return data