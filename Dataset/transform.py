
import numpy as np
import torch
import copy
from collections.abc import Mapping,Sequence
class Compose(object):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

def GridSample(
        data_dict,
        grid_size=0.05,
        keys=("coord",),#no need to sample vectors
        return_grid_coord=True,
    ):
        #print(f'input at the grid_sample{data_dict=}')
        scaled_coord = data_dict["coord"] / np.array(grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord #min_subtracted grid
        scaled_coord -= min_coord # min_subtracted scaled
        min_coord = min_coord * np.array(grid_size)# min is now the smallest point that has a grid location

        key = fnv_hash_vec(grid_coord) # assigning each grid location a random key
        idx_sort = np.argsort(key) # sorting asc based on keys
        key_sort = key[idx_sort] #key sort is sorted grid
        _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True) # getting the number of different grids
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1]) # this gives the starting index of the uniques
            + np.random.randint(0, count.max(), count.size) % count # this gives the index from the index sort to use
        )
        idx_unique = idx_sort[idx_select] # getting the unique indexes one from each grid
        if return_grid_coord:
            data_dict["grid_coord"] = grid_coord[idx_unique]
        for key in keys:
            data_dict[key] = data_dict[key][idx_unique]
        #print(f'output at the grid_sample{data_dict=}')
        return data_dict
    
def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
    

def GenerateOffsets(data_dict, keys=["coord","vectors","shape_id"], offset_keys_dict=dict(offset="coord")):
        #print(f'input_dict at the collect{data_dict}')
        for key, value in offset_keys_dict.items():
            data_dict[key] = torch.tensor([data_dict[value].shape[0]])
        return data_dict

def Shape_Consolidate(all_part_dicts,keys):
    shape_dict={}
    
    if 'part_coord' in keys:
        shape_dict['part_coord']  = torch.stack([d['part_coord'] for d in all_part_dicts])

        
    if 'obb_corners' in keys:
        shape_dict['obb_corners'] = torch.stack([d['obb_corners'] for d in all_part_dicts])
    
    if 'g_truth_axis' in keys:
        shape_dict['g_truth_axis']    = torch.stack([d['g_truth_axis'] for d in all_part_dicts])

    if 'motion_type' in keys:
        shape_dict['motion_type']    = torch.stack([d['motion_type'] for d in all_part_dicts])

    if 'orientation_label' in keys:
        shape_dict['orientation_label']    = torch.stack([d['orientation_label'] for d in all_part_dicts])

    if 'residual' in keys:
        shape_dict['residual']    = torch.stack([d['residual'] for d in all_part_dicts])
    
    if 'rotation_point' in keys:
        shape_dict['rotation_point'] = torch.stack([d['rotation_point'] for d in all_part_dicts])
    
    if 'centroid' in keys:
        shape_dict['centroid'] = torch.stack([d['centroid'] for d in all_part_dicts])

    if 'm' in keys:
        shape_dict['m'] = torch.stack([d['m'] for d in all_part_dicts])
    
    if 'paths' in keys:
        shape_dict['paths'] = all_part_dicts[0]['paths']
    
    if 'movable' in keys:
        shape_dict['movable'] = torch.tensor([d['movable'] for d in all_part_dicts],dtype=torch.bool)
    
    if 'is_cls' in keys:
        shape_dict['is_cls'] = torch.tensor([d['is_cls'] for d in all_part_dicts],dtype=torch.bool)

    if 'movable_id' in keys:
        shape_dict['movable_id'] = torch.tensor([d['movable_id'] for d in all_part_dicts],dtype=torch.int)

    if 'part_index' in keys:
        shape_dict['part_index'] = torch.tensor([d['part_index'] for d in all_part_dicts],dtype=torch.int)
    return shape_dict
    # for key in shape_dict:
    #     if key != 'paths':
    #         print(f"{key} {shape_dict[key].shape} {shape_dict[key].dtype}")
        

    
def ToTensor(data_dict,keys):
        #print(data_dict)
        if 'part_coord' in keys:
            data_dict['part_coord']    = torch.tensor(data_dict['part_coord']).float()
        if 'shape_coord' in keys:
            data_dict['shape_coord'] = torch.tensor(data_dict['shape_coord']).float()
        
        if 'obb_corners' in keys:
            data_dict['obb_corners'] = torch.tensor(data_dict['obb_corners']).float()
        
        if 'g_truth_axis' in keys:
            data_dict['g_truth_axis'] = torch.tensor(data_dict['g_truth_axis']).float()

        if 'motion_type' in keys:
            motion_type = torch.zeros(4,dtype=torch.float)
            motion_type[data_dict['motion_type']] = 1
            data_dict['motion_type'] = motion_type


        if 'orientation_label' in keys:
            data_dict['orientation_label'] = torch.from_numpy(data_dict['orientation_label'])
            orientation_label = torch.zeros(3,dtype=torch.float)
            orientation_label[data_dict['orientation_label']] = 1
            data_dict['orientation_label'] = orientation_label
        
        if 'residual' in keys:
            data_dict['residual'] = torch.tensor(data_dict['residual']).float()
        
        if 'rotation_point' in keys:
            data_dict['rotation_point'] = torch.tensor(data_dict['rotation_point']).float()

        
        if 'centroid' in data_dict.keys():
            data_dict['centroid'] = torch.tensor(data_dict['centroid']).float()

        if 'm' in data_dict.keys():
            data_dict['m'] = torch.tensor([data_dict['m']]).float()
        
        return data_dict


        

        # if isinstance(data_dict, torch.Tensor):
        #     return data_dict
        # elif isinstance(data_dict, str):
        #     # note that str is also a kind of sequence, judgement should before sequence
        #     return data_dict
        # elif isinstance(data_dict, int):
        #     return torch.LongTensor([data_dict])
        # elif isinstance(data_dict, float):
        #     return torch.FloatTensor([data_dict])
        # elif isinstance(data_dict, np.ndarray) and np.issubdtype(data_dict.dtype, bool):
        #     return torch.from_numpy(data_dict)
        # elif isinstance(data_dict, np.ndarray) and np.issubdtype(data_dict.dtype, np.integer):
        #     return torch.from_numpy(data_dict).long()
        # elif isinstance(data_dict, np.ndarray) and np.issubdtype(data_dict.dtype, np.floating):
        #     return torch.from_numpy(data_dict).float()
        # elif isinstance(data_dict, np.ndarray):
        #     return torch.from_numpy(data_dict)
        # elif isinstance(data_dict, Mapping):
        #     result = {sub_key: ToTensor(item) for sub_key, item in data_dict.items()}
        #     return result
        # elif isinstance(data_dict, list):#for vectors
        #     result = ToTensor(np.array(data_dict))
        #     return result
        # else:
        #     raise TypeError(f"type {type(data_dict)} cannot be converted to tensor.")
        # return 

#batch data should be per shape
def Point_TV3format(data_dict,keys):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    shape_dict = {}
    assert data_dict[0]['is_cls']==True

    if 'part_coord' in keys:
        shape_dict['part_coord'] = torch.cat([d['part_coord'] for d in data_dict],dim=0).float()

    if 'shape_coord' in keys:
        shape_dict['shape_dict'] = torch.cat([d['shape_coord'] for d in data_dict],dim=0).float()
    
    if 'obb_corners' in keys:
        shape_dict['obb_corners'] = torch.cat([d['obb_corners'] for d in data_dict],dim=0).float()
    
    if 'g_truth_axis' in keys:
        shape_dict['g_truth_axis'] = torch.cat([d['g_truth_axis'].unsqueeze(0) for d in data_dict],dim=0).float()

    if 'motion_type' in keys:
        shape_dict['motion_type'] = torch.cat([d['motion_type'].unsqueeze(0) for d in data_dict],dim=0).float()

    if 'orientation_label' in keys:
        shape_dict['orientation_label'] = torch.cat([d['orientation_label'].unsqueeze(0) for d in data_dict],dim=0).float()

    if 'residual' in keys:
        shape_dict['residual'] = torch.cat([d['residual'].unsqueeze(0) for d in data_dict],dim=0).float()
    
    if 'rotation_point' in keys:
        shape_dict['rotation_point'] = torch.cat([d['rotation_point'].unsqueeze(0) for d in data_dict],dim=0).float()

    shape_dict['num_parts'] = len(data_dict)
    return shape_dict
    # # elif isinstance(batch[0], str):
    # #     # str is also a kind of Sequence, judgement should before Sequence
    # #     return list(batch)
    # elif isinstance(batch[0], Mapping):
    #     batch_length = len(batch)
    #     batch = {key: Point_TV3format([d[key] for d in batch]) if key in keys else [d[key] for d in batch] for key in batch[0] }
    #     for key in batch.keys():
    #         if "offset" in key:
    #             batch[key] = torch.cumsum(batch[key], dim=0)
    #     # for key in batch:
    #     #     if(isinstance(batch[key],torch.Tensor)):
    #     #         print(f'{key=} {batch[key].shape=}')
    #     #     else:
    #     #         print(f'{key=} {len(batch[key])}')
    #     return batch


def filter_segments(data_dict, max_motion_vectors):
    vectors = data_dict['vectors']
    if(len(vectors)==0): # this is about handling the cls 
        return False
    positives = vectors[vectors[:,-1]==1]
    np.random.shuffle(positives)
    negatives = vectors[vectors[:,-1]==0]
    np.random.shuffle(negatives)
    min_number = min(len(positives),len(negatives))
    num_sample = min(min_number,max_motion_vectors)
    if(min_number< 1):
        return False ## takes care of non movable parts
    else:
        positives = positives[:num_sample]
        negatives = negatives[:num_sample]
        data_dict['vectors'] = np.concatenate((positives,negatives))
        np.random.shuffle(data_dict['vectors'])
        return True
        
    


def normalize_according_shape(data_dict,keys):
        centroid = np.mean(data_dict["shape_coord"], axis=0)
        data_dict["shape_coord"] -= centroid
        m = np.max(np.sqrt(np.sum(data_dict["shape_coord"] ** 2, axis=1)))
        data_dict["shape_coord"] = data_dict["shape_coord"] / m
        data_dict["part_coord"] -= centroid
        data_dict["part_coord"] = data_dict["part_coord"] / m
   
        data_dict["rotation_point"] -= centroid
        data_dict['rotation_point'] = data_dict['rotation_point']/m
        data_dict['centroid'] = centroid
        data_dict['m'] = m
        return data_dict

def Normalizeshape(data_dict,keys):
        # shape_coord
        if "shape_coord" in keys:
            centroid = np.mean(data_dict["shape_coord"], axis=0)
            data_dict["shape_coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["shape_coord"] ** 2, axis=1)))
            data_dict["shape_coord"] = data_dict["shape_coord"] / m

        if 'part_coord' in keys:
            centroid = np.mean(data_dict["part_coord"], axis=0)
            data_dict["part_coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["part_coord"] ** 2, axis=1)))
            data_dict["part_coord"] = data_dict["part_coord"] / m
        return data_dict
