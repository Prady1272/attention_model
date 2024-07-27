import os
import shutil
import argparse
import numpy as np
import torch
import math
torch.cuda.empty_cache()
import torch.nn as nn
from models.main import MultiTaskLoss,Predictor,Predictor_Transformer
import importlib
from Dataset.main import Mobility_Dataset
from Dataset.misc import custom_collate_fn
from utils.lr_sch import adjust_learning_rate
from utils.misc import axis_points,state_dict,loss_funs,compute_loss,compute_errors,fibonacci_sphere,compute_stats,compute_stats_transform,print_le
from torch.utils.tensorboard import SummaryWriter
import time, json
from tqdm import tqdm
from torch.utils.data import default_collate


def get_args():
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument('--base_dir',default='/home/pradyumngoya/working_dr')
    parser.add_argument('--data_root',default='snippets/data/partnet_mobility_root/')
    parser.add_argument('--use_detach',action='store_false',default=True)
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str, help="Path to retrieve latest checkpoint file relative to checkpoint folder")
    
    parser.add_argument('--resume_train',action='store_true',default=False)

    #output paths
    parser.add_argument('--base_output',default='./train_output')
    parser.add_argument('--checkpoint',default='checkpoint')
    parser.add_argument('--runs',default='runs')
    parser.add_argument('--logs',default='logs.txt')


    # model parameters
    parser.add_argument('--model',default='mlp')
    parser.add_argument('--loss_red',default='mean')
    parser.add_argument('--features_reduce',default='mean')
    parser.add_argument('--not_enable_flash',action='store_true',default=False)
    parser.add_argument('--obb_red',default='mean')
    parser.add_argument('--trained_checkpoint',required=True,type=str)
    parser.add_argument('--num_layers',default=2,type=int)

    #dataset parameters
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--max_seq_len", default=8, type=int, help="sequence lenth per item")
    parser.add_argument("--max_motion_vectors", default=1, type=int, help="motin_vectors per item")
    parser.add_argument("--num_workers", type=int,default=4)
    parser.add_argument("--loop", type=int,default=1)
    parser.add_argument("--category", type=str,required=True)
    

    # hyperameters of network/options for training
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--lr", default=5e-4, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=5e-5, type=float, help="Initial learning rate")
    parser.add_argument("--resume_epoch", default=0, type=int, help="Initialize the starting epoch")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Start from specified epoch number")
    
    
    parser.add_argument("--train_split_ratio", default=0.8, type=float, help="ratio of training split")
    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args()


# validation function
def predict(args,val_loader, model):
    
    model.eval()
    oriented_axis = fibonacci_sphere()
    for batch_data in tqdm(val_loader):
        with torch.no_grad():

            output = model(batch_data)

            axis,points = axis_points(oriented_axis,output)
            axis = axis.cpu().detach()
            points = points.cpu().detach()
            
            if (args.model=='transformer'):
                num_parts = output["num_parts"]-1
                axis,points = torch.split(axis,num_parts.tolist(),dim=0),torch.split(points,num_parts.tolist(),dim=0)
                for i, path in enumerate(output['paths']):
                    part_dict = torch.load(path)

                    assert len(part_dict)-1==axis[i].shape[0]
                    for j in range(len(part_dict)):
                        if(part_dict[j]["is_cls"]):
                            continue
                        part_dict[j]['predicted_axis'] = axis[i][k]
                        part_dict[j]['predicted_point'] = points[i][k]
                        k+=1
                        print(k)
                    torch.save(part_dict,path)
            elif(args.model=='mlp'):
                for i, path in enumerate(output['paths']):
                    part_dict = torch.load(path)
                    part_dict['predicted_axis'] = axis[i]
                    part_dict['predicted_point'] = points[i]
                    torch.save(part_dict,path)
          


    return 






def main(args):
    # taking care of model dependent variables
    args.data_root = os.path.join(args.data_root,args.model)
   #module = importlib.import_module(f'models.{args.model}')
    if args.model == "mlp":
        model = Predictor(enable_flash= not args.not_enable_flash,device = args.device,features_reduce=args.features_reduce)
    elif args.model == 'transformer':
        model = Predictor_Transformer(enable_flash= not args.not_enable_flash,device = args.device,
                features_reduce=args.features_reduce,
                num_layers=args.num_layers)

    if args.model == 'mlp':
        collate_fn = default_collate
    elif args.model == 'transformer':
        collate_fn = custom_collate_fn



    #making the output paths
    # Get the current local time
    if(args.trained_checkpoint!=''):
        checkpoint = args.trained_checkpoint
    else:
        print("add the checkpoint folder for visualization")
        exit()

    
    print(f'this is the checkpoint folder{checkpoint}')



    # initializing the models and the optimizers
    model.to(args.device)
   
    # for resuming training
    model_state,optimizer_state,epoch,best_loss = state_dict(os.path.join(checkpoint,args.resume_file))
    model.load_state_dict(model_state)
        

    #print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # initializing the datasets
    test_dataset = Mobility_Dataset(base_dir=args.base_dir,data_root=args.data_root, split='test',category=args.category,max_seq_len=args.max_seq_len, loop=1)
   
    print(f'{len(test_dataset)=}')


    
    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size*2,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True,)
    

    print("starting adding the results")
    predict(args,test_loader, model)
    print('finished adding back the data')
    
        



if __name__ == "__main__":
    
    args = get_args()
    main(args)