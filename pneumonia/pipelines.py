from pneumonia.utils import read_image
from pneumonia.utils import load_bboxes

# TODO AS: Cache image reading
def train_pipeline(cache, sample_db, image_path):
    image = read_image(image_path)
    bboxes = load_bboxes(sample_db, image_path)
    return image, bboxes

def validation_pipeline(cache, sample_db, image_path):
    image = read_image(image_path)
    bboxes = load_bboxes(sample_db, image_path)
    return image, bboxes
