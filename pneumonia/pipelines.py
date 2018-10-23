import numpy as np
from albumentations import (Resize, Normalize, Compose)

from pneumonia.utils import read_image
from pneumonia.utils import load_bboxes

def channels_first(image):
    return np.moveaxis(image, 2, 0)

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        return args

# TODO AS: Cache image reading
def train_pipeline(cache, sample_db, image_path):
    image = read_image(image_path)
    bboxes = load_bboxes(sample_db, image_path)
    args = Compose([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image, bboxes=bboxes)
    return args['image'], args.get('bboxes')

def validation_pipeline(cache, sample_db, image_path):
    image = read_image(image_path)
    bboxes = load_bboxes(sample_db, image_path)
    args = Compose([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst()
    ])(image=image, bboxes=bboxes)
    return args['image'], args.get('bboxes')
