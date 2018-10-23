import glob
from itertools import product

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch

def get_sample_split(sample_db, num_folds):
    np.random.seed(1991)
    # TODO AS: Find a way to use negative samples
    sample_db = sample_db[sample_db.x.notnull()]
    patient_ids = sample_db['patientId'].unique()
    fold_ids = np.random.randint(0, num_folds, len(patient_ids))
    return patient_ids, fold_ids

def get_sample_db(path):
    return pd.read_csv(path)

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.dcm'))

def read_image(path):
    image = pydicom.read_file(path).pixel_array
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    return image

def load_bboxes(sample_db, image_path):
    patient_id = image_path.split('/')[-1].split('.')[0]
    bboxes = sample_db[sample_db['patientId'] == patient_id]
    return bboxes[['x', 'y', 'width', 'height']].values

def generate_anchors(stride, scales, ratios, image_shape):
    max_y_shift = image_shape[0] // stride
    max_x_shift = image_shape[1] // stride

    anchors = []
    for y_shift, x_shift, scale, ratio in product(range(max_y_shift), range(max_x_shift), scales, ratios):
        x_center = stride / 2 + x_shift * stride - 1
        y_center = stride / 2 + y_shift * stride - 1
        width = scale * ratio
        height = scale / ratio
        anchors.append((
            x_center - width * 0.5 + 1,
            y_center - height * 0.5 + 1,
            x_center + width * 0.5,
            y_center + height * 0.5
        ))
    return np.array(anchors, dtype=np.float32)

def construct_deltas(gt_boxes, anchors):
    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1] + 1

    x_center_a = anchors[:, 0] + w_a * 0.5
    y_center_a = anchors[:, 1] + h_a * 0.5
    x_center_gt = gt_boxes[:, 0] + w_gt * 0.5
    y_center_gt = gt_boxes[:, 1] + h_gt * 0.5

    t_x = (x_center_gt - x_center_a) / w_a
    t_y = (y_center_gt - y_center_a) / h_a
    t_w = np.log(w_gt / w_a)
    t_h = np.log(h_gt / h_a)

    return np.column_stack((
        t_x,
        t_y,
        t_w,
        t_h
    )) / [0.1, 0.1, 0.2, 0.2]

def construct_boxes(deltas, anchors):
    deltas = deltas * [0.1, 0.1, 0.2, 0.2]
    t_x = deltas[:, 0]
    t_y = deltas[:, 1]
    t_w = deltas[:, 2]
    t_h = deltas[:, 3]

    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    x_center_a = anchors[:, 0] + w_a * 0.5
    y_center_a = anchors[:, 1] + h_a * 0.5

    w_gt = np.exp(t_w) * w_a
    h_gt = np.exp(t_h) * h_a

    x_center_gt = t_x * w_a + x_center_a
    y_center_gt = t_y * h_a + y_center_a

    x0 = x_center_gt - w_gt * 0.5
    y0 = y_center_gt - h_gt * 0.5
    x1 = x_center_gt + w_gt * 0.5 - 1
    y1 = y_center_gt + h_gt * 0.5 - 1

    return np.column_stack((
        x0,
        y0,
        x1,
        y1
    ))

def iou(bboxes_a, bboxes_b):
    tl = np.maximum(bboxes_a[:, None, :2], bboxes_b[:, :2])
    br = np.minimum(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
    area_i = np.prod(br - tl + 1, axis=2) * (tl <= br).all(axis=2)
    area_a = np.prod(bboxes_a[:, 2:] - bboxes_a[:, :2] + 1, axis=1)
    area_b = np.prod(bboxes_b[:, 2:] - bboxes_b[:, :2] + 1, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchors(stride, scales, ratios, image_shape):
    max_y_shift = image_shape[0] // stride
    max_x_shift = image_shape[1] // stride

    anchors = []
    for y_shift, x_shift, scale, ratio in product(range(max_y_shift), range(max_x_shift), scales, ratios):
        x_center = stride / 2 + x_shift * stride - 1
        y_center = stride / 2 + y_shift * stride - 1
        width = scale * ratio
        height = scale / ratio
        anchors.append((
            x_center - width * 0.5 + 1,
            y_center - height * 0.5 + 1,
            x_center + width * 0.5,
            y_center + height * 0.5
        ))
    return np.array(anchors, dtype=np.float32)

def generate_anchor_grid(scales, ratios, shape):
    anchors = []
    for scale, ratio, x_shift, y_shift in product(scales, ratios, range(shape[0]), range(shape[1])):
        x_center = (1 / shape[0]) * (x_shift + 0.5)
        y_center = (1 / shape[1]) * (y_shift + 0.5)
        width = scale * ratio
        height = scale / ratio
        anchors.append([
            x_center - width * 0.5,
            y_center - height * 0.5,
            x_center + width * 0.5,
            y_center + height * 0.5
        ])
    return np.array(anchors, dtype=np.float32)

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def from_numpy(obj):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(obj)
    else:
        return torch.FloatTensor(obj)

def to_numpy(tensor):
    return tensor.data.cpu().numpy()

if __name__ == '__main__':
    import pdb; pdb.set_trace()

