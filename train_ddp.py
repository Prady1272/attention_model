import os
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from attention_model.models.mlp import Predictor
from Dataset.main import Mobility_Dataset
from Dataset.misc import custom_collate_fn
from utils.lr_sch import adjust_learning_rate
from utils.misc import generate_paths,redirect_output_to_file,restore_original_output
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group,init_process_group

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

def get_args():
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument('--base_dir',default='/home/pradyumngoya/unity_data')
    parser.add_argument('--dataset_path', default="snippets/data/partnet_mobility_root/attention")
    parser.add_argument('--use_detach',action='store_false',default=True)
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str, help="Path to retrieve latest checkpoint file relative to checkpoint folder")
    parser.add_argument('--enable_flash',action='store_true',default=False)

    #output paths
    parser.add_argument('--base_output',default='./train_output')
    parser.add_argument('--checkpoint',default='checkpoint')
    parser.add_argument('--runs',default='runs')
    parser.add_argument('--logs',default='logs.txt')
    parser.add_argument('--redirect_logs',action='store_true',default=False)

    #dataset parameters
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for training")
    parser.add_argument("--max_seq_len", default=8, type=int, help="sequence lenth per item")
    parser.add_argument("--max_motion_vectors", default=1, type=int, help="motin_vectors per item")
    parser.add_argument("--num_workers", type=int,default=4)
    

    # hyperameters of network/options for training
    parser.add_argument("--epochs", default=400, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--lr", default=5e-3, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="Initial learning rate")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="Start from specified epoch number")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Start from specified epoch number")
    
    
    parser.add_argument("--train_split_ratio", default=0.8, type=float, help="ratio of training split")

    return parser.parse_args()

# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(train_loader, model, optimizer,additional_arguments):
    rank = additional_arguments['rank']
    model.train()  # switch to train mode
    loss_sum = 0.0
    loss_count = 0.0
    criteria = nn.BCELoss()

    print(f"[GPU{rank}]| Steps: {len(train_loader)}")

    for batch_data in ((train_loader)):
        

        optimizer.zero_grad()
        #print(" before foreward torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        output, label = model(batch_data)
        return
        #print(" after foreward torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        loss = criteria(output,label)
        loss.backward()
        #print(" after backward torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        optimizer.step()
        #print(" after optimizer torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))

        loss_sum +=(loss.cpu().detach().item())
        loss_count+=1

        last_time = time.time()
        #
        
        # ***********************************************************************

    return loss_sum / loss_count


# validation function
def val(val_loader, model,additional_arguments):
    print(f"[GPU{rank}]| Steps: {len(self.train_data)}")
    model.eval()
    loss_sum = 0.0
    loss_count = 0.0
    criteria = nn.BCELoss()
    for batch_data in (val_loader):
        with torch.no_grad():
            output, label = model(batch_data)
            loss = criteria(output,label)
            loss_sum +=(loss)
            loss_count+=1
        
        # ***********************************************************************

    return loss_sum / loss_count




def main(rank, world_size, args):
    #initializing the para meters for the training
    setup(rank,world_size)
   
    best_loss = 2e10
    best_epoch = -1


    #making the output paths
    # Get the current local time

    train_dataset = Mobility_Dataset(base_dir=args.base_dir, split='train',max_seq_len=args.max_seq_len,max_motion_vectors=args.max_motion_vectors)
    test_dataset = Mobility_Dataset(base_dir=args.base_dir, split='test',max_seq_len=args.max_seq_len,max_motion_vectors=args.max_motion_vectors)
    print(f'{len(train_dataset)=}')
    print(f'{len(test_dataset)=}')


    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    collate_fn=custom_collate_fn,
                    sampler=sampler
                )
    
    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size*2,
                    collate_fn=custom_collate_fn,)
    

    if(rank==0):
        checkpoint,runs,logs = generate_paths(args)
        writer = SummaryWriter(runs)
        if(args.redirect_logs):
            original_stdout,original_stderr,log_file = redirect_output_to_file(logs)
        


    print(args)

    #print(" before loading the model torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
    model = Predictor(device=rank,enable_flash=args.enable_flash)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    #print(" after loading the model torch.cuda.memory_allocated:%fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
    #print("=> Will use the (" + args.device + ") device.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))


    
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)  
        adjust_learning_rate(optimizer,epoch, args)
        train_loss = train(train_loader, model, optimizer,{'rank':rank})
        return
        if rank==0:
            val_loss = val(test_loader, model,{'rank',rank})
            is_best = val_loss<best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch
            
            save_checkpoint({"epoch": epoch + 1, "state_dict": model.module.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                            is_best, checkpoint_folder=checkpoint)
            print(f"Epoch {epoch+1:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch+1:d}. Best val loss: {best_loss:.8f}. \
                    Current lr : {optimizer.param_groups[0]['lr']:.8f}.")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
    if rank==0:
        writer.flush()
        writer.close()
        if(args.redirect_logs):
            restore_original_output(original_stdout,original_stderr,log_file)
    print('finished training')
    destroy_process_group()
    
        



if __name__ == "__main__":
    
    args = get_args()
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size,args),
        nprocs=world_size
    )