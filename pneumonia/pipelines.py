import numpy as np
from albumentations import (Resize, Normalize, Compose)
from albumentations.augmentations.bbox_utils import convert_bboxes_to_albumentations, convert_bboxes_from_albumentations

from pneumonia.utils import read_image
from pneumonia.utils import load_bboxes

def channels_first(image):
    return np.moveaxis(image, 2, 0)

class ConvertBboxesToAlbumentations:
    def __call__(self, **args):
        if args.get('bboxes') is None: return args
        args['bboxes'] = np.array(convert_bboxes_to_albumentations(args.get('bboxes'), 'coco', args['image'].shape[0], args['image'].shape[1]))
        return args

class ConvertBboxesToOriginal:
    def __call__(self, **args):
        if args.get('bboxes') is None: return args
        args['bboxes'] = np.array(convert_bboxes_from_albumentations(args.get('bboxes'), 'pascal_voc', args['image'].shape[1], args['image'].shape[2]))
        return args

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        return args

# TODO AS: Cache image reading
def train_pipeline(cache, sample_db, image_path):
    image = read_image(image_path)
    bboxes = load_bboxes(sample_db, image_path)
    args = Compose([
        ConvertBboxesToAlbumentations(),
        Resize(256, 256),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst(),
        ConvertBboxesToOriginal()
    ])(image=image, bboxes=bboxes)
    return args['image'], args.get('bboxes')

def validation_pipeline(cache, sample_db, image_path):
    image = read_image(image_path)
    bboxes = load_bboxes(sample_db, image_path)
    args = Compose([
        ConvertBboxesToAlbumentations(),
        Resize(256, 256),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ChannelsFirst(),
        ConvertBboxesToOriginal()
    ])(image=image, bboxes=bboxes)
    return args['image'], args.get('bboxes')
