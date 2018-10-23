import numpy as np
import torch
import torchvision
from tabulate import tabulate
from tqdm import tqdm

from pneumonia.utils import from_numpy
from pneumonia.utils import to_numpy
from pneumonia.utils import as_cuda

def fit_model(
        model,
        train_generator,
        validation_generator,
        optimizer,
        loss_fn,
        num_epochs,
        logger,
        callbacks=[],
        metrics=[]
    ):

    for epoch in tqdm(range(num_epochs)):
        num_batches = len(train_generator)
        logs = {}
        logs['train_loss'] = 0
        for func in metrics: logs[f'train_{func.__name__}'] = 0
        model.train()
        torch.set_grad_enabled(True)
        for callback in callbacks: callback.on_train_begin()
        for inputs, gt in tqdm(train_generator, total=num_batches):
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, gt)
            loss.backward()
            optimizer.step()
            logs['train_loss'] += loss.data[0]
            for func in metrics: logs[f'train_{func.__name__}'] += func(outputs.detach(), gt)
            for callback in callbacks: callback.on_train_batch_end(loss.data[0])

        logs['train_loss'] /= num_batches
        for func in metrics: logs[f'train_{func.__name__}'] /= num_batches

        logs['val_loss'] = 0
        for func in metrics: logs[f'val_{func.__name__}'] = 0
        all_outputs = []
        all_gt = []
        num_batches = len(validation_generator)
        model.eval()
        torch.set_grad_enabled(False)
        for inputs, gt in tqdm(validation_generator, total=num_batches):
            all_gt.append(gt)
            inputs, gt = from_numpy(inputs), from_numpy(gt)
            outputs = model(inputs)
            logs['val_loss'] += loss_fn(outputs, gt).data[0]
            for func in metrics: logs[f'val_{func.__name__}'] += func(outputs.detach(), gt)

            if isinstance(outputs, tuple):
                all_outputs.append(tuple(map(to_numpy, outputs)))
            else:
                all_outputs.append(to_numpy(outputs))
        logs['val_loss'] /= num_batches
        for func in metrics: logs[f'val_{func.__name__}'] /= num_batches

        if isinstance(all_outputs[0], tuple):
            all_outputs = list(map(np.concatenate, zip(*all_outputs)))
        else:
            all_outputs = np.concatenate(all_outputs)

        all_gt = np.concatenate(all_gt)
        for callback in callbacks: callback.on_validation_end(logs, all_outputs, all_gt)

        epoch_rows = [['epoch', epoch]]
        for name, value in logs.items():
            epoch_rows.append([name, f'{value:.3f}'])

        logger(tabulate(epoch_rows))
