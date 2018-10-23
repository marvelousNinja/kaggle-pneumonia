import glob

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

