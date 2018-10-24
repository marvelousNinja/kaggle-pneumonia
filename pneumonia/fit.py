import torch
import numpy as np
import matplotlib #; matplotlib.use('agg')
from fire import Fire

from pneumonia.callbacks.model_checkpoint import ModelCheckpoint
from pneumonia.callbacks.model_checkpoint import load_checkpoint
from pneumonia.callbacks.lr_on_plateau import LROnPlateau
from pneumonia.generators import get_train_generator
from pneumonia.generators import get_validation_generator
from pneumonia.loggers import make_loggers
from pneumonia.losses import compute_loss
from pneumonia.models.retinanet import Retinanet
from pneumonia.training import fit_model
from pneumonia.utils import as_cuda

def fit(
        num_epochs=100,
        limit=None,
        validation_limit=None,
        batch_size=2,
        lr=.005,
        checkpoint_path=None,
        telegram=False,
        visualize=False,
        num_folds=5,
        train_fold_ids=[0, 1, 2, 3],
        validation_fold_ids=[4]
    ):
    torch.backends.cudnn.benchmark = True
    np.random.seed(1991)
    logger, image_logger = make_loggers(telegram)

    if checkpoint_path:
        model = load_checkpoint(checkpoint_path)
    else:
        model = Retinanet(1, 3)

    model = as_cuda(model)
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr, weight_decay=1e-3, momentum=0.9, nesterov=True)
    train_generator = get_train_generator(num_folds, train_fold_ids, batch_size, limit)
    callbacks = [
        ModelCheckpoint(model, 'linknet', 'val_loss', 'min', logger),
        # CyclicLR(step_size=len(train_generator) * 2, min_lr=0.0001, max_lr=0.005, optimizer=optimizer, logger=logger),
        # LRSchedule(optimizer, [(0, 0.003), (2, 0.01), (12, 0.001), (17, 0.0001)], logger),
        # LRRangeTest(0.00001, 1.0, 20000, optimizer, image_logger),
        LROnPlateau('val_loss', optimizer, mode='min', factor=0.5, patience=4, min_lr=0, logger=logger),
        # ConfusionMatrix([0, 1], logger)
    ]

    if visualize:
        callbacks.extend([
            #LearningCurve(['train_loss', 'val_loss', 'train_mean_iou', 'val_mean_iou', 'train_mean_ap', 'val_mean_ap'], image_logger),
            #PredictionGrid(80, image_logger, mean_iou),
            #LossSurface(image_logger, loss_surface_fn),
            #Histogram(image_logger, mean_iou),
            #WeightGrid(model, image_logger, 32)
        ])

    fit_model(
        model=model,
        train_generator=train_generator,
        validation_generator=get_validation_generator(num_folds, validation_fold_ids, batch_size, validation_limit),
        optimizer=optimizer,
        loss_fn=compute_loss,
        num_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        metrics=[]
    )

def prof():
    import profile
    import pstats
    profile.run('fit(batch_size=4, limit=100, num_epochs=1)', 'fit.profile')
    stats = pstats.Stats('fit.profile')
    stats.sort_stats('cumulative').print_stats(30)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    Fire()
