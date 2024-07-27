import os
import random
from collections import defaultdict
import sys
import torch
import json
import argparse

def min_shapes(shapes_data_dir,category,all_cat_shapes):
        # motion_type,axis->shapes
        shapes_matrix = [[set() for i in range(3)] for j in range(3)]
        # shapes-> motion_type,axis
        shape_motions_dict = defaultdict()
        for shape in all_cat_shapes:
            shape_motions = [[0 for i in range(3)] for j in range(3)]
            input_data = torch.load(os.path.join(shapes_data_dir,"_".join([category,shape])+".pth"))
            for data in input_data:# looping through all the parts
                if(data['motion_type']==3):
                    continue
                shapes_matrix[data['motion_type']][data['label'][0]].add(shape)
                shape_motions[data['motion_type']][data['label'][0]] = 1
            shape_motions_dict[shape] = shape_motions

        shapes_matrix_debug= [[0 for i in range(3)] for j in range(3)]
        for motion_type in range(3):
            for orientation_type in range(3):
                shapes_matrix_debug[motion_type][orientation_type] = len(shapes_matrix[motion_type][orientation_type])

        training_shapes = set() # all training shapes
        done_motions = [[0 for i in range(3)] for j in range(3)] # seeing all done_motions
        for motion_type in range(3):
            for orientation_type in range(3):
                if(len(shapes_matrix[motion_type][orientation_type])>0 and done_motions[motion_type][orientation_type]==0): # checking the if the motion should be covered and if not covered
                    new_shape = random.sample(shapes_matrix[motion_type][orientation_type],1)[0]
                    training_shapes.add(new_shape) # adding the shape to the training shapes
                    new_shape_motions = shape_motions_dict[new_shape] # the motions of the new shapes
                    for new_motion_type in range(3):
                        for new_orientation_type in range(3):
                            if new_shape_motions[new_motion_type][new_orientation_type]==1: # checking if the new shape has that motions
                                done_motions[new_motion_type][new_orientation_type]=1 # the done_motions =1
        # for debugging purposes
        # print('*'*10)
        # print(f'{shapes_matrix_debug=}')
        # print(f'{done_motions=}')
        # print(f'{training_shapes=}')
        # print(f'{shapes_matrix=}')
        return training_shapes
    

def split_directories(dataset_path, training_data_dir, train_ratio=0.6,split_train_val=0.8,use_semi_split=True):
    assert train_ratio<=0.8
    shapes_data_dir = os.path.join(dataset_path,training_data_dir)
    shapes_data = os.listdir(shapes_data_dir)
    cat_shape_dict = defaultdict(list)
    semi_train_file = 'train.struct.json'
    semi_test_file = 'test.struct.json'

    # if not using the semi split remove this
    # this code is just initializing the directory. 
    if use_semi_split:
        semi_train_shapes = None
        semi_test_shapes = None
        with open (semi_train_file,'r') as file:
            semi_train_shapes = json.load(file)
        with open(semi_test_file,'r') as file:
            semi_test_shapes = json.load(file)

        # categories have to have the first letter capital, in the files given that was not the case
        update_categories=[]
        for category in semi_train_shapes:
            if category[0].islower():
                update_categories.append(category)
        
        print(update_categories)

        for category in update_categories:
                newcategory = category
                newcategory = newcategory[0].upper()+newcategory[1:]
                semi_train_shapes[newcategory] = semi_train_shapes[category]
                semi_test_shapes[newcategory] = semi_test_shapes[category]

    # sorting them by category.
    for cat_shape in shapes_data:
        try:
            if 'test' in cat_shape or 'train' in cat_shape or 'val' in cat_shape:
                raise Exception
            [category, fname] = cat_shape.split('_')
            fname = fname.split(".")[0]
            cat_shape_dict[category].append(fname)
        
        except Exception as e:
            print(cat_shape)
            print(e)
            continue

  
    train_split,val_split,test_split = defaultdict(list),defaultdict(list),defaultdict(list)
    for category in cat_shape_dict:
        #  this first gets the training shapes and then from the training shapes gets the val index
        category_train_shapes = list(set(cat_shape_dict[category]) & set(semi_train_shapes[category]))
        random.shuffle(category_train_shapes)
        split_val_index = int(len(category_train_shapes) * split_train_val)
    
        for shape in category_train_shapes[:split_val_index]:
            train_split[category].append("_".join([category,shape])+".pth")

        for shape in category_train_shapes[split_val_index:]:
            val_split[category].append("_".join([category,shape])+".pth")
        # just gets the test split
        category_test_shapes = list(set(cat_shape_dict[category]) & set(semi_test_shapes[category]))
        print(f'{category}  {split_val_index} {len(category_train_shapes)-split_val_index} {len(category_test_shapes)}')
        for shape in category_test_shapes:
            test_split[category].append("_".join([category,shape])+".pth")


    # writing to a json file
    with open(os.path.join(shapes_data_dir, 'train_new.json'), 'w') as f_train:
        json.dump(train_split,f_train,indent=4)
    
    with open(os.path.join(shapes_data_dir, 'test_new.json'), 'w') as f_test:
        json.dump(test_split,f_test,indent=4)

    
    with open(os.path.join(shapes_data_dir, 'val_new.json'), 'w') as f_val:
        json.dump(val_split,f_val,indent=4)
    

    train_split,val_split,test_split = defaultdict(list),defaultdict(list),defaultdict(list)
    for category,all_cat_shapes in cat_shape_dict.items():
        
        training_shapes = min_shapes(shapes_data_dir=shapes_data_dir,category=category,all_cat_shapes=all_cat_shapes)
        test_val_shapes = [d  for d in all_cat_shapes if d not in training_shapes ]
        val_shapes = min_shapes(shapes_data_dir=shapes_data_dir,category=category,all_cat_shapes=test_val_shapes)
        test_shapes = [d  for d in test_val_shapes if d not in val_shapes ]

        for shape in training_shapes:
            train_split[category].append("_".join([category,shape])+".pth")
        
        for shape in val_shapes:
            val_split[category].append("_".join([category,shape])+".pth")
        
        for shape in test_shapes:
            test_split[category].append("_".join([category,shape])+".pth")

        print(f'{training_shapes=}')
        print(f'{test_shapes=}')
        print(f'{val_shapes=}')
        print(len(training_shapes),len(val_shapes),len(test_shapes))
        
    with open(os.path.join(shapes_data_dir, 'train_few.json'), 'w') as f_train:
        json.dump(train_split,f_train,indent=4)
    
    with open(os.path.join(shapes_data_dir, 'test_few.json'), 'w') as f_test:
        json.dump(test_split,f_test,indent=4)

    
    with open(os.path.join(shapes_data_dir, 'val_few.json'), 'w') as f_val:
        json.dump(val_split,f_val,indent=4)
    
            

                

    
        # for shape in category_train_shapes[:split_val_index]:
        #     train_split[category].append("_".join([category,shape])+".pth")

        # for shape in category_train_shapes[split_val_index:]:
        #     val_split[category].append("_".join([category,shape])+".pth")
        # # just gets the test split
        # category_test_shapes = list(set(cat_shape_dict[category]) & set(semi_test_shapes[category]))
        # print(f'{category}  {split_val_index} {len(category_train_shapes)-split_val_index} {len(category_test_shapes)}')
        # for shape in category_test_shapes:
        #     test_split[category].append("_".join([category,shape])+".pth")

    
    

    

  

    
    

# Usage example:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process base directory and dataset path.')
    parser.add_argument('--base_dir', type=str, default='/home/pradyumngoya/working_dr', 
                        help='Base directory path')
    parser.add_argument('--dataset_path', type=str, default='snippets/data/partnet_mobility_root', 
                        help='Dataset path')

    parser.add_argument('--training_data_dir', type=str, default='fine_transformer_mobilities', 
                        help='training_data')

    args = parser.parse_args()

    base_dir = args.base_dir
    dataset_path = args.dataset_path

    split_directories(os.path.join(base_dir, dataset_path),args.training_data_dir, train_ratio=0.8,use_semi_split=True)





    # train_split_parts, test_split_parts = defaultdict(list), defaultdict(list)
    # print("going over the parts")
    # for category in cat_shape_dict:
    #     print(category)
    #     for train_shape in train_split[category]:
    #         for part in os.listdir(os.path.join(shapes_data_dir,train_shape)):
    #             if(part == "cls_.pth"):
    #                 continue
    #             else:
    #                 data_dict = torch.load(os.path.join(shapes_data_dir, train_shape, part))
    #                 train_split_parts[category].append('|'.join([train_shape,part]))

    #     for test_shape in test_split[category]:
    #         for part in os.listdir(os.path.join(shapes_data_dir,test_shape)):
    #             if(part == "cls_.pth"):
    #                 continue
    #             else:
    #                 data_dict = torch.load(os.path.join(shapes_data_dir, test_shape, part))
    #                 test_split_parts[category].append('|'.join([test_shape,part]))


    # Save the directory names to train.txt and test.txt