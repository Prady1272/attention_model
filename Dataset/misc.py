import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch_list):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """

    
    batch_dict = {}
    keys = batch_list[0].keys()

    if 'part_coord' in keys:
        batch_dict['part_coord'] = torch.cat([d['part_coord'] for d in batch_list],dim=0)

    if 'shape_coord' in keys:
        batch_dict['shape_dict'] = torch.cat([d['shape_coord'] for d in batch_list],dim=0)
    
    if 'obb_corners' in keys:
        batch_dict['obb_corners'] = torch.cat([d['obb_corners'] for d in batch_list],dim=0)
    
    if 'g_truth_axis' in keys:
        batch_dict['g_truth_axis'] = torch.cat([d['g_truth_axis'] for d in batch_list],dim=0)

    if 'motion_type' in keys:
        batch_dict['motion_type'] = torch.cat([d['motion_type'] for d in batch_list],dim=0)

    if 'orientation_label' in keys:
        batch_dict['orientation_label'] = torch.cat([d['orientation_label'] for d in batch_list],dim=0)

    if 'residual' in keys:
        batch_dict['residual'] = torch.cat([d['residual'] for d in batch_list],dim=0)
    
    if 'rotation_point' in keys:
        batch_dict['rotation_point'] = torch.cat([d['rotation_point'] for d in batch_list],dim=0)

    if 'centroid' in keys:
        batch_dict['centroid'] = torch.cat([d['centroid'] for d in batch_list],dim=0)

    if 'm' in keys:
        batch_dict['m'] = torch.cat([d['m'] for d in batch_list],dim=0)
    
    if 'movable' in keys:
        batch_dict['movable'] = torch.cat([d['movable'] for d in batch_list],dim=0)

    if 'movable_id' in keys:
        batch_dict['movable_id'] = torch.cat([d['movable_id'] for d in batch_list],dim=0)

    if 'part_index' in keys:
        batch_dict['part_index'] = torch.cat([d['part_index'] for d in batch_list],dim=0)
    if 'is_cls' in keys:
        batch_dict['is_cls'] = torch.cat([d['is_cls'] for d in batch_list],dim=0)
    
    #changed
    if 'paths' in keys:
        batch_dict['paths'] = [d['paths'] for d in batch_list]


    batch_dict['num_parts'] = torch.tensor([d['part_coord'].shape[0] for d in batch_list],dtype=torch.long)
    
    batch_dict['split_indices'] = torch.cumsum(batch_dict['num_parts'],dim=0,dtype=torch.long)


    
    return batch_dict

    # if not isinstance(batch, Sequence):
    #     raise TypeError(f"{batch.dtype} is not supported.")

    # if isinstance(batch[0], torch.Tensor):
    #     return torch.cat(list(batch))
    # elif isinstance(batch[0], str):
    #     # str is also a kind of Sequence, judgement should before Sequence
    #     return list(batch)
    # elif isinstance(batch[0], Sequence):
    #     for data in batch:
    #         data.append(torch.tensor([data[0].shape[0]]))
    #     batch = [custom_collate_fn(samples) for samples in zip(*batch)]
    #     batch[-1] = torch.cumsum(batch[-1], dim=0).int()
    #     return batch
    # elif isinstance(batch[0], Mapping):
    #     batch = {key: custom_collate_fn([d[key] for d in batch]) for key in batch[0]}
    #     for key in batch.keys():
    #         if "offset" in key:
    #             batch[key] = torch.cumsum(batch[key], dim=0)
    #     return batch