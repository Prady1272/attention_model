"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from .transform import GridSample,Normalizeshape,normalize_according_shape,ToTensor,Point_TV3format,filter_segments,Shape_Consolidate
from functools import partial
import random
import json
import copy


class Mobility_Dataset(Dataset):
    def __init__(
        self,
        split="train",
        base_dir="/home/pradyumngoya/unity_data",
        data_root="snippets/data/partnet_mobility_root/fine_tune_sorted_4",
        #the first should be normalizecoor and the last should be Merge
        transform=(normalize_according_shape,ToTensor),
        max_seq_len = 20,
        split_index = 0,
        max_motion_vectors = 5,
        loop=4,
        num_points=2048,
        category='Box',
        max_shapes=int(10e8),
    ):
        super(Mobility_Dataset, self).__init__()
        self.data_path = os.path.join(base_dir,data_root)
        print(f'{self.data_path=}')
        self.split = split
        self.split_index = split_index
        self.transform = transform
        self.loop = loop
        self.max_seq_len = max_seq_len
        #self.max_motion_vectors = max_motion_vectors
        self.category = category
        self.num_points = num_points
        self.max_shapes = max_shapes
        assert max_seq_len>=2
        # self.cache = cache
        # self.loop = (
        #     loop if not test_mode else 1
        # )  # force make loop = 1 while in test mode
        #self.test_mode = test_mode
        #self.test_cfg = test_cfg if test_mode else None

        # if test_mode:
        #     self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
        #     self.test_crop = (
        #         TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
        #     )
        #     self.post_transform = Compose(self.test_cfg.post_transform)
        #     self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        # logger = get_root_logger()
        # logger.info(
        #     "Totally {} x {} samples in {} set.".format(
        #         len(self.data_list), self.loop, split
        #     )
        # )
    
    def get_data_list_helper(self):
        files_path = os.path.join(self.data_path,self.split+f'_{self.split_index}.json')
        with open(files_path, 'r') as file:
            data = json.load(file)
        if self.category != 'all': 
            data_list = data[self.category]
        else:
            data_list = []
            for category,item in data.items():
                data_list.extend(item)
        return sorted(data_list)[:self.max_shapes]


    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = self.get_data_list_helper()
        else:
            raise Exception
        return data_list

    
 

    def prepare_data_mlp(self, idx):
        part = self.data_list[idx % len(self.data_list)]
        shape,part = part.split("|")
        path  =  os.path.join(self.data_path,shape,part)
        
        data = torch.load(path)
        part_coord = np.array(data["points"])
        part_indices = np.random.choice(part_coord.shape[0],self.num_points,replace=False)
        part_coord = part_coord[part_indices]
        obb_corners = np.array(data['obb_corners'])
        motion_type = data['motion_type']
        if type(motion_type ) == list:
            motion_type = motion_type[0]
        orinetation_label = int(data['label'])
        g_truth_axis = np.array(data['g_truth_axis'],dtype=float)
        residual = data['residual']
        rotation_point = np.array(data['g_truth_point'],dtype=float)
        shape_id = path.split(os.path.sep)[-2]
        shape_data = torch.load(os.path.join(self.data_path,shape_id,'cls.pth'))
        shape_coord = shape_data['points']
        shape_indices = np.random.choice(shape_coord.shape[0],self.num_points,replace=False)
        shape_coord = shape_coord[shape_indices]
        data_dict = dict(
            part_coord = part_coord,
            shape_coord = shape_coord,
            obb_corners = obb_corners,
            g_truth_axis = g_truth_axis,
            motion_type = motion_type,
            orientation_label = orinetation_label,
            residual = residual,
            rotation_point = rotation_point,
            paths=path
        )
        data_dict = self.apply_transform(data_dict,list(data_dict.keys()))
        return data_dict
    
    def prepare_data_transformer(self, idx):
        shape = self.data_list[idx % len(self.data_list)]
        path  =  os.path.join(self.data_path,shape)
        all_part_dicts = []
        input_data = torch.load(path,map_location='cpu')
        # just for the cls
        cls_index = -1
        for i, part_dict in enumerate(input_data):
            if part_dict['is_cls']:
                if cls_index!=-1:
                    print("two cls")
                    raise KeyError
                cls_index = i
            part_dict['part_index'] = i
        if cls_index==-1:
            print(f"{shape} does not have a cls points")
            raise KeyError

        cls_dict = input_data.pop(cls_index)
        input_data = random.sample(input_data,min(len(input_data),self.max_seq_len))
        input_data.insert(0,cls_dict)
        if len(input_data)<=1:
            # print(f'{path=}')
            # print("this shape had only 1 part")
            return self.prepare_data_transformer(random.randint(0,len(self)-1))

       

        for data in input_data:
            shape_coord = copy.deepcopy(np.array(input_data[0]['points']))
            part_coord = np.array(data["points"])
            part_indices = np.random.choice(part_coord.shape[0],self.num_points,replace=False)
            part_coord = part_coord[part_indices]
            obb_corners = np.array(data['obb_corners'])
            motion_type = data['motion_type']
            if type(motion_type ) == list:
                motion_type = motion_type[0]
            orientation_label = np.array(data['label'],dtype=int)
            g_truth_axis_com = np.array(data['g_truth_axis'],dtype=float).reshape((-1,3))
            residual_com = np.array(data['residual'],dtype=float).reshape((-1,3))

            is_cls = data['is_cls']
            rotation_point_com = np.array(data['g_truth_point'],dtype=float).reshape((-1,3))


            g_truth_axis = np.zeros(shape=(3,3))
            residual = np.zeros(shape=(3,3))
            rotation_point = np.zeros(shape=(3,3))
            # print('in the data loader')
            # print(f'{orientation_label=}')
            # print(f'{g_truth_axis_com=}')
            # print(f'{residual_com=}')

            for i,label in enumerate(orientation_label):
                g_truth_axis[label] = g_truth_axis_com[i]
                residual[label] = residual_com[i]
                rotation_point[label] = rotation_point_com[i]
            
            # print(f'{g_truth_axis=}')
            # print(f'{residual=}')
            # print("done witht the data loader")



            
            data_dict = dict(
                movable=data['movable'],
                part_coord=part_coord,
                shape_coord = shape_coord,
                obb_corners = obb_corners,
                g_truth_axis = g_truth_axis,
                motion_type = motion_type,
                orientation_label = orientation_label,
                residual = residual,
                rotation_point = rotation_point,
                is_cls=is_cls,
                paths=path,
                movable_id=int(data['movable_id']),
                part_index = data["part_index"]
            )
            all_part_dicts.append(self.apply_transform(data_dict,keys= list(data_dict.keys())))

        
        return Shape_Consolidate(all_part_dicts,all_part_dicts[0].keys())
    
    def apply_transform(self,data_dict,keys):
        for transform in self.transform:
            data_dict = transform(data_dict=data_dict,keys=keys)
        return data_dict



    def __getitem__(self, idx):
        path_separator = os.path.sep
        final_dir = self.data_path.split(path_separator)[-1]
        return self.prepare_data_transformer(idx)
            # self[random.randint(0,len(self)-1)]
        # if 'pretrain' in final_dir:
        #     return self.prepare_data_transformer(idx)
        #     return self.prepare_data_transformer(idx)
        # elif 'mlp' in final_dir:
        #     return self.prepare_data_mlp(idx)
        

    def __len__(self):
        return len(self.data_list) * self.loop
