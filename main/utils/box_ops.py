# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch


def box_area(boxes):
    return boxes[:, 1] - boxes[:, 0]


def box_cxw_to_x(x):
    x_c, w = x.unbind(-1)
    b = [(x_c - 0.5 * w),
         (x_c + 0.5 * w)]
    return torch.stack(b, dim=-1)


def box_x_to_cxw(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2,
         (x1 - x0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,1]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,1]

    wh = (rb - lt).clamp(min=0)  # [N,M,1]
    inter = wh[:, :, 0]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    The boxes should be in [x0, x1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0]

    return iou - (area - union) / area
