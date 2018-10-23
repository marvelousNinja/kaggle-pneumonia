import math
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np

from pneumonia.pipelines import train_pipeline
from pneumonia.pipelines import validation_pipeline
from pneumonia.utils import get_images_in
from pneumonia.utils import get_sample_db
from pneumonia.utils import get_sample_split

class DataGenerator:
    def __init__(self, records, batch_size, transform, shuffle=True, drop_last=False):
        self.records = records
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle: np.random.shuffle(self.records)
        batch = []
        pool = ThreadPool()
        prefetch_size = 2000
        num_slices = len(self.records) // prefetch_size + 1

        for i in range(num_slices):
            start = i * prefetch_size
            end = start + prefetch_size
            for output in map(self.transform, self.records[start:end]):
                batch.append(output)
                if len(batch) >= self.batch_size:
                    split_outputs = list(zip(*batch))
                    merged_outputs = []
                    for split_output in split_outputs:
                        # TODO AS: Will not work with 1 sample in the batch
                        if split_output[0].shape[0] == 1:
                            enumerated_samples = []
                            for i, sample in enumerate(split_output):
                                enumerated_samples.extend(np.hstack([np.tile([i], (len(sample), 1)), sample]))
                            merged_outputs.append(np.stack(enumerated_samples))
                        else:
                            merged_outputs.append(np.stack(split_output))

                    yield list(map(np.stack, merged_outputs))
                    batch = []

        if (not self.drop_last) and len(batch) > 0:
            split_outputs = list(zip(*batch))
            merged_outputs = []
            for split_output in split_outputs:
                # TODO AS: Will not work with 1 sample in the batch
                if split_output[0].shape[0] == 1:
                    enumerated_samples = []
                    for i, sample in enumerate(split_output):
                        enumerated_samples.extend(np.hstack([np.tile([i], (len(sample), 1)), sample]))
                    merged_outputs.append(np.stack(enumerated_samples))
                else:
                    merged_outputs.append(np.stack(split_output))
            yield list(map(np.stack, merged_outputs))

        pool.close()

    def __len__(self):
        num_batches = len(self.records) / self.batch_size
        if self.drop_last:
            return math.floor(num_batches)
        else:
            return math.ceil(num_batches)

def get_validation_generator(num_folds, fold_ids, batch_size, limit=None):
    sample_db = get_sample_db('data/stage_1_train_labels.csv')
    all_image_ids, all_fold_ids = get_sample_split(sample_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]
    image_paths = list(map(lambda id: f'data/train/{id}.dcm', image_ids))
    transform = partial(validation_pipeline, {}, sample_db)
    return DataGenerator(image_paths[:limit], batch_size, transform, shuffle=False, drop_last=True)

def get_train_generator(num_folds, fold_ids, batch_size, limit=None):
    sample_db = get_sample_db('data/stage_1_train_labels.csv')
    all_image_ids, all_fold_ids = get_sample_split(sample_db, num_folds)
    image_ids = all_image_ids[np.isin(all_fold_ids, fold_ids)]
    image_paths = list(map(lambda id: f'data/train/{id}.dcm', image_ids))
    transform = partial(train_pipeline, {}, sample_db)
    return DataGenerator(image_paths[:limit], batch_size, transform, drop_last=True)

def get_test_generator(batch_size, limit=None):
    sample_db = get_sample_db('data/stage_1_train_labels.csv')
    image_paths = get_images_in('data/test')
    transform = partial(validation_pipeline, {}, sample_db)
    return DataGenerator(image_paths[:limit], batch_size, transform, shuffle=False)

if __name__ == '__main__':
    generator = get_train_generator(10, [0, 1], 16)
    batch = next(generator.__iter__())