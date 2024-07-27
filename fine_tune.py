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


def get_args():
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument('--base_dir',default='/home/pradyumngoya/unity_data',help="base directory to connect all other paths")
    parser.add_argument('--data_root', default="snippets/data/partnet_mobility_root/fine_tune",help="should hold the data root folder that contains .pth")
    #parser.add_argument('--base_model',default='Partnet/Unsupervised/attention_model/train_output/2024-05-05_23_23/checkpoint/model_best.pth.tar')
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str, help="model to retrieve the model")
    parser.add_argument('--enable_flash',action='store_true',default=False)
    parser.add_argument('--resume_train',action='store_false',default=True)

    #output paths
    parser.add_argument('--base_output',default='./train_output',help="all the outputs")
    parser.add_argument("--fine_folder", default="fine_tune/", type=str, help="Folder to save fine_tuned model")
    parser.add_argument('--checkpoint',default='checkpoint',help="checkpoints for the base model")
    parser.add_argument('--runs',default='runs_fine_tune',help="for tensorboard")
    parser.add_argument('--logs',default='logs.txt')
    parser.add_argument('--redirect_logs',action='store_true',default=False)

    #dataset parameters
    parser.add_argument("--batch_size", default=24, type=int, help="Batch size for training")
    parser.add_argument("--max_seq_len", default=8, type=int, help="sequence lenth per item")
    parser.add_argument("--max_motion_vectors", default=1, type=int, help="motin_vectors per item")
    parser.add_argument("--num_workers", type=int,default=4)
    parser.add_argument("--loop", type=int,default=1)
    

    # hyperameters of network/options for training
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--lr", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="Initial learning rate")
    parser.add_argument("--resume_epoch", default=0, type=int, help="Initialize the starting epoch")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Start from specified epoch number")

    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args()

# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(train_loader, model, optimizer):
    model.train()  # switch to train mode
    loss_sum = 0.0
    loss_count = 0.0
    criteria = nn.BCELoss()

    for batch_data in tqdm((train_loader)):

        optimizer.zero_grad()
        #print(" before foreward torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        torch.cuda.reset_peak_memory_stats(0)
        output, label = model(batch_data)
        #print(" after foreward torch.cuda.memory_reservexd: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))
        #print(" after foreward torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        loss = criteria(output,label)
        torch.cuda.reset_peak_memory_stats(0)
        loss.backward()
        #print(" after backward torch.cuda.memory_reservexd: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))
        # print(" after backward torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))
        optimizer.step()
        #print(" after optimizer torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/(1024**3)))



        loss_value = loss.cpu().detach().item()
        loss_sum +=(loss_value)
        loss_count+=1
        del loss
        #tqdm.write(f"Loss: {loss_value}")
        
        # ***********************************************************************

    return loss_sum / loss_count


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
    #initializing the para meters for the training
   
    best_loss = 2e10
    best_epoch = -1


    #making the output paths
    # Get the current local time

   
    checkpoint,runs,logs,fine_folder = retrieve_paths(args,fine_tune=True)
    if(args.redirect_logs):
        original_stdout,original_stderr,log_file = redirect_output_to_file(logs)
    writer = SummaryWriter(runs)

    model = Predictor(enable_flash=args.enable_flash,device = args.device)
    model.to(args.device)
    
    print(f'loading from the base file {checkpoint}')
    model_state,optimizer_state,epoch,best_loss = state_dict(checkpoint)
    model.load_state_dict(model_state)

    print("loaded the base model")

    #freezing some layers of the mode
    unfrozen_layer = ['final_projection']

    for name, param in model.named_parameters():
        if not any(layer in name for layer in unfrozen_layer):
            param.requires_grad = False
    

    #for debugging
    # for name, param in model.named_parameters():
    #     print(f"{name} is {'trainable' if param.requires_grad else 'frozen'}")
    

    # for the optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))


   
    train_dataset = Mobility_Dataset(base_dir=args.base_dir, 
                                     data_root=args.data_root,
                                     split='train',
                                     max_seq_len=args.max_seq_len,
                                     max_motion_vectors=args.max_motion_vectors,
                                     loop=args.loop)
    test_dataset = Mobility_Dataset(base_dir=args.base_dir,
                                    data_root=args.data_root,
                                    split='test',
                                    max_seq_len=args.max_seq_len,
                                    max_motion_vectors=args.max_motion_vectors,
                                    loop=args.loop)
    print(f'{len(train_dataset)=}')
    print(f'{len(test_dataset)=}')
    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=custom_collate_fn,
                    pin_memory=True,
                    drop_last=True,
                    persistent_workers=True,
                )
    
    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=custom_collate_fn,
                    pin_memory=True,
                    persistent_workers=True,)

    print("starting training")
    for epoch in range(args.resume_epoch,args.epochs):
        adjust_learning_rate(optimizer,epoch, args)
        train_loss = train(train_loader, model, optimizer)
        if(epoch%5==0):
            val_loss,accuracy = val(test_loader, model)
            is_best = val_loss<best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch
            
            save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                            is_best, checkpoint_folder=fine_folder)
            print(f"Epoch {epoch+1:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. accuracy {accuracy}. Best Epoch: {best_epoch+1:d}. Best val loss: {best_loss:.8f}. \
                    Current lr : {optimizer.param_groups[0]['lr']:.8f}.")
            writer.add_scalar("Loss/val", val_loss, epoch)
        else:
            print(f"Epoch {epoch+1:d}. train_loss: {train_loss:.8f}. Current lr : {optimizer.param_groups[0]['lr']:.8f}.")
            save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                            is_best=False, checkpoint_folder=fine_folder)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
    
    writer.flush()
    writer.close()
    if(args.redirect_logs):
        restore_original_output(original_stdout,original_stderr,log_file)
    print('finished training')
    
        



if __name__ == "__main__":
    
    args = get_args()
    main(args)