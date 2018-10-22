from datetime import datetime
from functools import partial

import torch

from pneumonia.utils import as_cuda
from pneumonia.callbacks.callback import Callback

def save_checkpoint(model, path):
    torch.save(model, path)

def load_checkpoint(path):
    if torch.cuda.is_available():
        return as_cuda(torch.load(path))
    else:
        return torch.load(path, map_location='cpu')

def generate_checkpoint_path(prefix, timestamp, epoch, value):
    name = f'{prefix}-{timestamp}-{epoch:02d}-{value:.5f}.pt'
    return f'./data/models/{name}'

class ModelCheckpoint(Callback):
    def __init__(self, model, prefix, log_to_track, mode, logger=None):
        self.epoch = 0
        self.model = model
        self.logger = logger
        self.mode = mode
        self.value = float('inf') if mode == 'min' else 0.0
        self.log_to_track = log_to_track
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
        self.generate_checkpoint_path = partial(generate_checkpoint_path, prefix, timestamp)

    def on_validation_end(self, logs, outputs, gt):
        value = logs[self.log_to_track]
        if self.mode == 'min':
            update_needed = self.value > value
        else:
            update_needed = self.value < value

        if update_needed:
            checkpoint_path = self.generate_checkpoint_path(self.epoch, value)
            save_checkpoint(self.model, checkpoint_path)
            self.value = value
            if self.logger: self.logger(f'Checkpoint saved {checkpoint_path}')
        self.epoch += 1
