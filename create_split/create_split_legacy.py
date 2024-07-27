import os
import random

def split_directories(base_dir, train_ratio=0.80,save_path=None):
    subdirs = [d for d in os.listdir(base_dir)]
    random.shuffle(subdirs)
    
    # Calculate the split index
    split_index = int(len(subdirs) * train_ratio)
    print(f'{split_index=}')
    
    # Split into training and testing sets
    train_dirs = subdirs[:split_index]
    test_dirs = subdirs[split_index:]
    
    # Save the directory names to train.txt and test.txt


    with open(os.path.join(save_path, 'train.txt'), 'w') as f_train:
        for dir_name in train_dirs:
            f_train.write(dir_name + '\n')
    
    with open(os.path.join(save_path, 'test.txt'), 'w') as f_test:
        for dir_name in test_dirs:
            f_test.write(dir_name + '\n')

# Usage example:
if __name__ == '__main__':
    base_dir = '/home/pradyumngoya/unity_data'
    # base_dir = '/project/pi_ekalogerakis_umass_edu/pgoyal/'
    done_path = 'snippets/data/partnet_mobility_root/fine_done'
    split_directories(os.path.join(base_dir,done_path),train_ratio=0.8)
