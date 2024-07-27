import os
import shutil
import argparse
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.backends.cudnn as cudnn
from attention_model.models.mlp import Predictor
from Dataset.main import Mobility_Dataset
from Dataset.misc import custom_collate_fn
from utils.lr_sch import adjust_learning_rate
from utils.misc import generate_paths,redirect_output_to_file,restore_original_output,retrieve_paths,state_dict
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import time
from collections import defaultdict
import json


def get_args():
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument('--base_dir',default='/home/pradyumngoya/working_dr',help="base directory to connect all other paths")
    parser.add_argument('--data_root', default="snippets/data/partnet_mobility_root/fine_tune_sorted_4",help="should hold the data root folder that contains .pth")
    #parser.add_argument('--base_model',default='Partnet/Unsupervised/attention_model/train_output/2024-05-05_23_23/checkpoint/model_best.pth.tar')
    parser.add_argument("--resume_file", default="checkpoint.pth.tar", type=str, help="model to retrieve the model")
    parser.add_argument('--enable_flash',action='store_true',default=False)
    parser.add_argument('--resume_train',action='store_false',default=True)

    #output paths
    parser.add_argument('--base_output',default='./train_output',help="all the outputs")
    parser.add_argument("--fine_folder", default="fine_tune/", type=str, help="Folder to save fine_tuned model")
    parser.add_argument('--checkpoint',default='fine_tune',help="checkpoints for the fine tuned")
    parser.add_argument('--runs',default='runs_fine_tune',help="for tensorboard")
    parser.add_argument('--logs',default='logs.txt')
    parser.add_argument('--redirect_logs',action='store_true',default=False)

    #dataset parameters
    parser.add_argument("--batch_size", default=24, type=int, help="Batch size for training")
    parser.add_argument("--max_seq_len", default=8, type=int, help="sequence lenth per item")
    parser.add_argument("--max_motion_vectors", default=1, type=int, help="motin_vectors per item")
    parser.add_argument("--num_workers", type=int,default=4)
    parser.add_argument("--loop", type=int,default=1)
    parser.add_argument("--max_instances", type=int,default=100)
    parser.add_argument("--category", default='Chair', type=str, help="category split to compute the stats")
    

    # hyperameters of network/options for training
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--lr", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="Initial learning rate")
    parser.add_argument("--resume_epoch", default=0, type=int, help="Initialize the starting epoch")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Start from specified epoch number")

    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args()





# validation function
def val(val_loader, model):
    model.eval()
    loss_sum = 0.0
    loss_count = 0.0
    criteria = nn.BCELoss()
    #for accuracy
    total_correct=0
    total=0
    for batch_data in tqdm(val_loader):
        with torch.no_grad():
            output, label = model(batch_data)
            loss = criteria(output,label)
            loss_sum +=(loss)
            loss_count+=1

            #for accuracy
            tp = (output > 0.5) & (label == 1.0)
            tn = (output < 0.5) & (label == 0.0)
            correct = tp | tn
            correct = correct.sum().item()
            # print(output,label,correct)
            total_correct += correct
            total+=len(label)

            
        
        # ***********************************************************************

    return loss_sum / loss_count,total_correct/total




def main(args):
    with open (os.path.join(args.base_dir, args.data_root,'train.json'),'r') as file:
        data = json.load(file)
    category_paths = data[args.category]
    all_paths = []
    for path in category_paths:
        shape, part = path.split("|")
        all_paths.append(os.path.join(args.base_dir,args.data_root,shape,part))



    label_bin = [0 for _ in range (16)]
    type_bin = [0 for _ in range(4)]

    for path in all_paths:
        part_dict = torch.load(path)
        motion_type = part_dict['motion_type']
        if type(motion_type) ==list:
            motion_type = motion_type[0]
        type_bin[motion_type]+=1
        if motion_type != 3:
            label_bin[part_dict['label']]+=1

    print(f'{label_bin=}')
    print(f'{type_bin=}')

        
                    





                
        
        
def calculate_axis_error(p_drt,t_drt):
    drt_cos = np.sum(p_drt * t_drt, axis=1) / (np.linalg.norm(p_drt, axis=1) * np.linalg.norm(t_drt))
    drt_error = np.rad2deg(np.arccos(drt_cos))
    drt_error[drt_error > 90] = 180 - drt_error[drt_error > 90]
    return drt_error


def calculate_point_error(p_pos,t_drt,t_pos):
    pos_error = np.linalg.norm(np.cross(p_pos - t_pos, t_drt) / np.linalg.norm(t_drt),axis=1)
    return pos_error




if __name__ == "__main__":
    
    args = get_args()
    main(args)