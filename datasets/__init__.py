# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .vid_multi import build as build_vid_multi
from .vid_single import build as build_vid_single
from .vid_multi_eval import build as build_vid_multi_eval


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    if args.dataset_file == 'vid_single':
        return build_vid_single(image_set, args)
    if args.dataset_file == "vid_multi":
        return build_vid_multi(image_set, args)
    if args.dataset_file == "visual":
        return build_vid_multi(image_set, args)    
    if args.dataset_file == "vid_multi_eval":
        return build_vid_multi_eval(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
