import numpy as np
import torch
from torch.nn.functional import l1_loss
from torch.nn.functional import binary_cross_entropy_with_logits

from pneumonia.utils import from_numpy
from pneumonia.utils import to_numpy
from pneumonia.utils import construct_deltas
from pneumonia.utils import iou
from pneumonia.utils import generate_anchor_grid

def compute_loss(outputs, gt):
    loss = 0
    gt = to_numpy(gt)
    for i, (pred_logits, pred_deltas) in enumerate(zip(*outputs)):
        import pdb; pdb.set_trace()
        gt_boxes = gt[gt[:, 0] == i][:, 1:]

        anchors = []
        ratios = [0.5, 1.0, 2.0]
        for shape_x, shape_y, scale in [(64, 64, 1/64), (32, 32, 1/32), (16, 16, 1/16), (8, 8, 1/8)]:
            anchors.extend(generate_anchor_grid([scale], ratios, (shape_x, shape_y)))
        anchors = np.array(anchors) * 256
        true_labels, true_deltas = generate_targets(anchors, gt_boxes, (256, 256))
        sample_loss = 0

        positive = (true_labels == 1).nonzero().view(-1)
        if len(positive) > 0:
            sample_loss += l1_loss(pred_deltas[positive], true_deltas[positive])

        not_ignored = (true_labels != -1).nonzero().view(-1)
        if len(not_ignored) > 0:
            sample_loss += focal_loss(pred_logits[not_ignored], true_labels[not_ignored])
        loss += sample_loss
    return loss / len(outputs[0])

def generate_targets(anchors, gt_boxes, image_shape, positive_threshold=0.5, negative_threshold=0.4):
    ious = iou(anchors, gt_boxes)
    # For each anchor row, find maximum along columns
    gt_indicies = np.argmax(ious, axis=1)

    # For each anchor row, actual values of maximum IoU
    max_iou_per_anchor = ious[range(len(anchors)), gt_indicies]

    # For each gt box, anchor with the highest IoU (including ties)
    max_iou_per_gt_box = np.max(ious, axis=0)
    anchors_with_max_iou, gt_boxes_for_max_anchors = np.where(ious == max_iou_per_gt_box)

    # While anchor has max IoU for some GT box, it may overlap with other GT box better
    anchors_with_max_iou = anchors_with_max_iou[max_iou_per_anchor[anchors_with_max_iou] == max_iou_per_gt_box[gt_boxes_for_max_anchors]]

    # Anchors what cross image boundary
    outside_image = np.where(~(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] < image_shape[1]) &
        (anchors[:, 3] < image_shape[0])
    ))[0]

    # Negative: 0, Positive: 1, Neutral: -1
    neutral = -1,
    positive = 1
    negative = 0

    labels = np.repeat(neutral, len(anchors))
    labels[max_iou_per_anchor < negative_threshold] = negative
    labels[anchors_with_max_iou] = positive
    labels[max_iou_per_anchor >= positive_threshold] = positive
    labels[outside_image] = neutral

    deltas = construct_deltas(gt_boxes[gt_indicies], anchors)
    return from_numpy(labels), from_numpy(deltas)

def focal_loss(logits, labels, average=True):
    bce = binary_cross_entropy_with_logits(logits, labels[:, None], reduction='none')
    probs = torch.sigmoid(logits)
    signs = 2 * labels.float() - 1
    diffs = labels.float() - signs * probs
    losses = bce * (diffs ** 2)
    if average: return losses.mean()
    return losses