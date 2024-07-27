import os
import random
from collections import defaultdict
import sys
import torch
import json
def split_directories(dataset_path, train_ratio=0.8,save_path=None,num_parts=100):
    assert train_ratio<=0.8
    done_dir = os.path.join(dataset_path,'fine_done')
    shapes_data_dir = os.path.join(dataset_path,'mlp')
    done_shapes = set([d for d in os.listdir(done_dir)])
    shapes_data = [d for d in os.listdir(shapes_data_dir)]
    cat_shape_dict = defaultdict(list)
    semi_train_file = 'train.struct.json'
    semi_test_file = 'test.struct.json'

    # if not using the semi split remove this
    semi_train_shapes = None
    semi_test_shapes = None
    with open (semi_train_file,'r') as file:
        semi_train_shapes = json.load(file)
    with open(semi_test_file,'r') as file:
        semi_test_shapes = json.load(file)

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

    for cat_shape in shapes_data:
        try:
            category, fname = cat_shape.split('_')
            # shape_path = os.path.join(shapes_data_dir,cat_shape)
            # parts_list = torch.load(shape_path)
            # print(len(parts_list))
            if fname in done_shapes:
                cat_shape_dict[category].append(fname)
        except:
            continue
    # use this if not using semi split
    # train_split,val_split,test_split = defaultdict(list),defaultdict(list),defaultdict(list)
    # for category in cat_shape_dict:
    #     split_index = int(len(cat_shape_dict[category]) * train_ratio)
    #     split_test_index = int(len(cat_shape_dict)*(train_ratio+0.1))
    #     print(f'{category} {split_index} {len(cat_shape_dict[category])}')
    #     train_split[category] = cat_shape_dict[category][:split_index]
    #     val_split[category] = cat_shape_dict[category][split_index:split_test_index]
    #     test_split[category] = cat_shape_dict[category][split_index:split_test_index]

    #do not use this if not using semi split
    train_split,val_split,test_split = defaultdict(list),defaultdict(list),defaultdict(list)
    for category in cat_shape_dict:
        category_train_shapes = list(set(cat_shape_dict[category]) & set(semi_train_shapes[category]))
       
        for shape in category_train_shapes:
            train_split[category].append("_".join([category,shape]))

        category_test_shapes = list(set(cat_shape_dict[category]) & set(semi_test_shapes[category]))
        print(f'{category}  {len(category_train_shapes)} {len(category_test_shapes)}')
        for shape in category_test_shapes:
            test_split[category].append("_".join([category,shape]))

    

    train_split_parts, test_split_parts,val_split_parts = defaultdict(list), defaultdict(list),defaultdict(list)
    print("going over the parts")
    for category in cat_shape_dict:
        print(category)
        for train_shape in train_split[category]:
            for part in os.listdir(os.path.join(shapes_data_dir,train_shape)):
                print(part)
                if(part == "cls.pth"):
                    continue
                else:
                    #data_dict = torch.load(os.path.join(shapes_data_dir, train_shape, part))
                    train_split_parts[category].append('|'.join([train_shape,part]))

        for test_shape in test_split[category]:
            for part in os.listdir(os.path.join(shapes_data_dir,test_shape)):
                if(part == "cls.pth"):
                    continue
                else:
                    #data_dict = torch.load(os.path.join(shapes_data_dir, test_shape, part))
                    test_split_parts[category].append('|'.join([test_shape,part]))
        
        for val_shape in val_split[category]:
            for part in os.listdir(os.path.join(shapes_data_dir,val_shape)):
                if(part == "cls.pth"):
                    continue
                else:
                    #data_dict = torch.load(os.path.join(shapes_data_dir, val_shape, part))
                    val_split_parts[category].append('|'.join([val_shape,part]))

  

    # writing to a json file
    with open(os.path.join(shapes_data_dir, 'train.json'), 'w') as f_train:
        json.dump(train_split_parts,f_train,indent=4)
    
    with open(os.path.join(shapes_data_dir, 'test.json'), 'w') as f_test:
        json.dump(test_split_parts,f_test,indent=4)

    
    with open(os.path.join(shapes_data_dir, 'val.json'), 'w') as f_val:
        json.dump(val_split_parts,f_val,indent=4)
    

# Usage example:
if __name__ == '__main__':
    base_dir = '/home/pradyumngoya/working_dr'
    # base_dir = '/project/pi_ekalogerakis_umass_edu/pgoyal/'
    dataset_path = 'snippets/data/partnet_mobility_root/'
    split_directories(os.path.join(base_dir,dataset_path),train_ratio=0.8)





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