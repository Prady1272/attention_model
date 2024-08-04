import os
import csv, random
import shutil
import argparse
import numpy as np
import torch
import wandb
import math
torch.cuda.empty_cache()
import torch.nn as nn
from models.main import MultiTaskLoss,Predictor_Transformer
import importlib
from Dataset.main import Mobility_Dataset
from Dataset.misc import custom_collate_fn
from utils.lr_sch import adjust_learning_rate
from utils.misc import retrieve_paths,state_dict,loss_funs,compute_loss,compute_stats,compute_stats_transform,print_le
from utils.misc import generate_pretraining_paths, generate_fine_tuning_paths, vis_helper,attn_weight_images
from torch.utils.tensorboard import SummaryWriter
import time, json
from tqdm import tqdm
from torch.utils.data import default_collate





def max_number_parts(tensor):
    indices = torch.nonzero(tensor > 0, as_tuple=True)
    largest_index = indices[0][-1].item()
    return largest_index



def get_args():
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust
    parser.add_argument('--base_dir',default='/home/pradyumngoya/working_dr')
    parser.add_argument('--data_root',default='snippets/data/partnet_mobility_root')
    parser.add_argument('--split_index',type=int,default=0)
    #output paths
    parser.add_argument('--base_output',default='./train_output')
    parser.add_argument('--checkpoint',default='checkpoint')
    parser.add_argument('--runs',default='runs')
    
    
    #resume_training or fine tuning, if resume training always retrives the latest path, testing for visualization
    parser.add_argument('--resume_train',action='store_true',default=False)
    parser.add_argument('--pretraining',action='store_true')
    parser.add_argument("--test",action='store_true')

    


    # model parameters
    parser.add_argument('--model',default='transformer')
    parser.add_argument('--use_multi_loss',action="store_false")
    parser.add_argument('--loss_reduction',default='mean')
    parser.add_argument('--features_reduce',default='mean')
    parser.add_argument('--enable_flash',action='store_false',default=False)
    parser.add_argument('--obb_red',default='mean')
    parser.add_argument('--encode_part',action='store_false')
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--negative_slope", default=0.2, type=float)
    parser.add_argument('--encode_shape',action='store_false')
    parser.add_argument('--num_layers',default=1,type=int)
    parser.add_argument('--use_type_weights',action='store_false')
    parser.add_argument('--use_orientation_weights',action='store_false')
    parser.add_argument("--best_model", default="model_best.pth.tar", type=str, help="Path to retrieve best model")

    #dataset parameters
    parser.add_argument("--batch_size", default=5, type=int, help="Batch size for training")
    parser.add_argument("--max_seq_len", default=30, type=int, help="sequence lenth per item")
    #parser.add_argument("--max_motion_vectors", default=1, type=int, help="motin_vectors per item")
    parser.add_argument("--num_workers", type=int,default=4)
    parser.add_argument("--loop", type=int,default=1)
    parser.add_argument("--max_shapes",default=int(10e8),type=int)
    parser.add_argument("--category",type=str,default='Bottle')
    parser.add_argument("--category_index", type=int,default=0)
    

    # hyperameters of network/options for training
    parser.add_argument("--epochs", default=400, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--lr", default=5e-4, type=float, help="Initial learning rate")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="Initial learning rate")
    parser.add_argument("--resume_epoch", default=0, type=int, help="Initialize the starting epoch")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="Start from specified epoch number")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Start from specified epoch number")
    
    
    #parser.add_argument("--train_split_ratio", default=0.8, type=float, help="ratio of training split")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_seed",action="store_true")
    

    return parser.parse_args()

# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):

    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))





def run_all(run_type,additional_arguments,data_loader, model,multitaskloss_instance, optimizer,
          motion_lfn,orientation_lfn,residual_lfn,point_lfn,vis=False):
    args = additional_arguments['args']
    loss_weights = additional_arguments['loss_weights']

    if run_type == "train":
        model.train()  # switch to train mode
        if args.use_multi_loss:
            multitaskloss_instance.train()
    else:
        model.eval()
        if args.use_multi_loss:
            multitaskloss_instance.eval()


    loss_sum = 0.0
    count = 0

    type_accuracy_sum = []
    orientation_error_sum = []
    point_error_sum = []

    type_loss_sum =  0
    orientation_loss_sum = 0
    residual_loss_sum = 0
    point_loss_sum = 0

    for batch_data in tqdm((data_loader),disable=True):


        optimizer.zero_grad()



        #torch.cuda.reset_peak_memory_stats(0)
        if run_type=="train":
            output,attn_output_weights = model(batch_data)

            #print(" after foreward torch.cuda.memory_reservexd: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))
            (type_loss, orientation_loss, residual_loss, point_loss),(accuracy,orientation_error,point_error), details = compute_loss(args,output,motion_lfn,orientation_lfn,residual_lfn,point_lfn)
            losses = torch.stack([type_loss,orientation_loss,residual_loss,point_loss])
            if args.use_multi_loss:
                losses = torch.stack([type_loss,orientation_loss,residual_loss,point_loss])
                loss = multitaskloss_instance(losses)
            else:
                loss = (losses*loss_weights).sum()



            #torch.cuda.reset_peak_memory_stats(0)
            loss.backward()
            optimizer.step()

            wandb.log({"loss":loss.cpu().detach().item()})
            #print(" after backward torch.cuda.memory_reservexd: %fGB"%(torch.cuda.max_memory_reserved(0)/(1024**3)))
        else:
            with torch.no_grad():
                output,attn_output_weights = model(batch_data)
                (type_loss, orientation_loss, residual_loss, point_loss),(accuracy,orientation_error,point_error), details = compute_loss(args,output,motion_lfn,orientation_lfn,residual_lfn,point_lfn)
                losses = torch.stack([type_loss,orientation_loss,residual_loss,point_loss])
                if args.use_multi_loss:
                    losses = torch.stack([type_loss,orientation_loss,residual_loss,point_loss])
                    loss = multitaskloss_instance(losses)
                else:
                    loss = (losses*loss_weights).sum()





        


        
        loss_sum += (loss.cpu().detach().item())
        # loss_sum+=0
        type_loss_sum += (type_loss.cpu().detach().item())
        orientation_loss_sum += (orientation_loss.cpu().detach().item())
        residual_loss_sum += (residual_loss.cpu().detach().item())
        point_loss_sum += (point_loss.cpu().detach().item())
        count += 1

        if not args.pretraining:
            type_accuracy_sum.extend(accuracy.cpu().detach().tolist())
            orientation_error_sum.extend(orientation_error.cpu().detach().tolist())
            point_error_sum.extend(point_error.cpu().detach().tolist())


        
        if run_type=="test" and vis: # all part index tell which index the current part is in the list of parts. all_index just keeps a track from all the shapes
            details = details+(orientation_error,point_error)
            vis_helper(args,details)
                
                           
    return ((loss_sum / count, type_loss_sum/count, orientation_loss_sum/count, residual_loss_sum/count,point_loss_sum/count), 
            (sum(type_accuracy_sum)/(len(type_accuracy_sum) if len(type_accuracy_sum) >0 else 1),
             sum(orientation_error_sum)/(len(orientation_error_sum) if len(orientation_error_sum)>0 else 1),sum(point_error_sum)/(len(point_error_sum) if len(point_error_sum) >0 else 1)))






def load_models(args,checkpoint,model,optimizer,multitaskloss_instance=None):
    model_state,optimizer_state,epoch,best_loss = state_dict(checkpoint)
    args.resume_epoch = epoch
    best_epoch = epoch
    best_loss = best_loss
    if "state_dict" in model_state:
        model.load_state_dict(model_state["state_dict"])
        if (model_state['multi_loss_dict'] is not None) and (multitaskloss_instance is not None):
            assert not args.pretraining
            multitaskloss_instance.load_state_dict(model_state['multi_loss_dict'])
    optimizer.load_state_dict(optimizer_state)


    return best_epoch, best_loss

def main(args):
    #initializing the para meters for the training
   
    best_loss = 2e10
    best_train_loss = 2e10
    best_epoch = -1
    

    print(args)
    args.base_output += "_ep" if args.encode_part else "" 
    args.base_output += '_es'+str(args.num_layers) if args.encode_shape else ""

    # taking care of model dependent variables
    if args.pretraining:
        args.data_root = os.path.join(args.data_root,f'pretrain_transformer_mobilities')
        args.epochs = 200
    else:
        args.data_root = os.path.join(args.data_root,f'fine_transformer_mobilities')

    # module = importlib.import_module(f'models.{args.model}')
    # if args.model == "mlp":
    #     model = Predictor(enable_flash= not args.not_enable_flash,device = args.device,features_reduce=args.features_reduce)
    # elif args.model == 'transformer':
    model = Predictor_Transformer(enable_flash= args.enable_flash,device = args.device,num_layers=args.num_layers,encode_part=args.encode_part, encode_shape=args.encode_shape,
                                dropout=args.dropout,negative_slope=args.negative_slope)
    # else:
    #     print("model not supported")
    #     exit()


    if args.use_multi_loss:
        is_regression = torch.Tensor([False, False, True,True]).to(args.device)
        multitaskloss_instance = MultiTaskLoss(is_regression,reduction=args.loss_reduction)
    else:
        multitaskloss_instance = None

    if args.model == 'mlp':
        collate_fn = default_collate
    elif args.model == 'transformer':
        collate_fn = custom_collate_fn

    # remove this
    for temp_split_index in range(3):
        motion_type_bins, orientation_bins,num_parts = compute_stats_transform(args,split='train',split_index=temp_split_index)
        print(f'train {temp_split_index=} {motion_type_bins=} {orientation_bins=}')
        
    for temp_split_index in range(3):
        motion_type_bins, orientation_bins,num_parts = compute_stats_transform(args,split='val',split_index=temp_split_index)
        print(f'val {temp_split_index=} {motion_type_bins=} {orientation_bins=}')
    
    for temp_split_index in range(3):
        motion_type_bins, orientation_bins,num_parts = compute_stats_transform(args,split='test',split_index=temp_split_index)
        print(f'test {temp_split_index =} {motion_type_bins=} {orientation_bins=}')


    # taking care of bins
    if args.model=='transformer':
        motion_type_bins, orientation_bins,num_parts = compute_stats_transform(args)
        if not args.pretraining:
            args.max_seq_len = max_number_parts(torch.tensor(num_parts))
            args.batch_size = max(int(math.floor(220/args.max_seq_len)),1) # depends on the true sequence length
        else:
            args.loop =  max(int(math.floor(max_number_parts(torch.tensor(num_parts))/args.max_seq_len)),1) # depends on the working sequence length
    elif args.model == 'mlp':
        motion_type_bins, orientation_bins = compute_stats(args)
    
    print(f'{motion_type_bins=} {orientation_bins=} {max_number_parts(torch.tensor(num_parts))=} {args.max_seq_len=} {args.batch_size=} {args.loop=} ')
    # motion_type_bins = torch.tensor([number if number>1 else 1 for number in motion_type_bins],device=args.device,dtype=torch.float)
    # orientation_bins = torch.tensor([number if number>1 else 1 for number in orientation_bins],device=args.device,dtype=torch.float)
    motion_type_bins = torch.tensor(motion_type_bins,device=args.device,dtype=torch.float)
    orientation_bins = torch.tensor(orientation_bins,device=args.device,dtype=torch.float)
   
    motion_weights  = motion_type_bins.sum()/motion_type_bins
    orientation_weights = orientation_bins.sum()/orientation_bins
    
    print("before clipping")
    print(f'{motion_weights=}')
    print(f'{orientation_weights=}')
    motion_weights = motion_weights.clip(max=10000)
    for i,m in enumerate(motion_weights):
        if m==10000:
            motion_weights[i]=0
    orientation_weights = orientation_weights.clip(max=10000)
    for i,m in enumerate(orientation_bins):
        if m==10000:
            orientation_bins[i]=0
    print("before normalization")
    print(f'{motion_weights=}')
    print(f'{orientation_weights=}')
    motion_weights = motion_weights/torch.linalg.norm(motion_weights)
    orientation_weights = orientation_weights/torch.linalg.norm(orientation_weights)

    if not args.use_type_weights:
        motion_weights = torch.tensor([1,1,1,1],dtype=torch.float,device=args.device)

    if not args.use_orientation_weights:
        orientation_weights = torch.tensor([1,1,1],dtype=torch.float,device=args.device)

    if args.pretraining:
        loss_weights = torch.tensor([0.2,1,1,1]).to(args.device)
    else:
        loss_weights = torch.tensor([1,1,1,1]).to(args.device)

    print(f'{loss_weights=}')
    print(f'{multitaskloss_instance=}')
    print(f'{motion_weights=}')
    print(f'{orientation_weights=}')
    motion_lfn,orientation_lfn,residual_lfn,point_lfn = loss_funs(args,motion_weights=motion_weights,
                                                                  orientation_type_weights=orientation_weights)

    # if args.model=='transformer':
    #     indices = np.where(np.array(num_parts) > 0)[0]
    #     largest_index = indices.max()
    #     args.batch_size = max(math.floor(92/largest_index),1)


    #making the output paths
    # Get the current local time
    if(not args.resume_train and args.pretraining): # starting from scratch pretraining case 1
        checkpoint,runs,csv_file = generate_pretraining_paths(args)
        print("its starting from scratch Pretraining")
        print(f'{checkpoint=}')

    elif(args.resume_train or args.test): # takes care of case 2 and 4 and 5. (4 and 5 same path will be retrieved)
        checkpoint,runs,csv_file = retrieve_paths(args) # just retrieves the latest path according to pretraining
        print("its resuming")
        print(f'{checkpoint=}')
    
    elif not args.pretraining: # this is fine tuning
        checkpoint, runs,csv_file = generate_fine_tuning_paths(args)
        print('fine tuning')
        print(f'{checkpoint=}')

    else:
        assert False, "wrong arguments"





    # initializing the models and the optimizers
    writer = SummaryWriter(runs)
    model.to(args.device)
    if args.use_multi_loss:
        multitaskloss_instance.to(args.device)

    if args.use_multi_loss:
        params = list(model.parameters()) + list(multitaskloss_instance.parameters())
    else:
        params = list(model.parameters()) 

    # params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # for resuming training
    if(args.resume_train or args.test):
        best_epoch,best_loss = load_models(checkpoint=checkpoint,model=model,optimizer=optimizer,
                                           multitaskloss_instance=multitaskloss_instance)
       
        print("loaded the model correctly")
        

    #print("=> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # initializing the datasets
    train_dataset = Mobility_Dataset(base_dir=args.base_dir,data_root=args.data_root, split='train',split_index = args.split_index,category=args.category,max_seq_len=args.max_seq_len, loop=args.loop,max_shapes=args.max_shapes)
    test_dataset = Mobility_Dataset(base_dir=args.base_dir,data_root=args.data_root, split='test',split_index = args.split_index,category=args.category,max_seq_len=args.max_seq_len, loop=1)
    val_dataset = Mobility_Dataset(base_dir=args.base_dir,data_root=args.data_root, split='val',split_index = args.split_index,category=args.category,max_seq_len=args.max_seq_len, loop=1)
    print(f'{len(train_dataset)=}')
    print(f'{len(test_dataset)=}')
    print(f'{len(val_dataset)=}')
    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True if args.num_workers != 0 else False,
                )
    
    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=args.batch_size*2,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True if args.num_workers != 0 else False,)
    

    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size*2,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True if args.num_workers != 0 else False,)
    
    print('dataloaders are working well')
    additional_arguments = {'args':args,'loss_weights':loss_weights}
    if (args.test):
        test_losses,test_errors = run_all(run_type="test",additional_arguments=additional_arguments,data_loader=test_loader, model=model,multitaskloss_instance=multitaskloss_instance,optimizer=optimizer,
                                                        motion_lfn=motion_lfn,orientation_lfn=orientation_lfn,residual_lfn=residual_lfn,point_lfn=point_lfn,vis=True)
        print_le(writer,epoch,optimizer,losses=test_losses,errors=test_errors,split="test",best_epoch=best_epoch)
        exit()
    print("starting training")
    val_step = 5
    if args.pretraining:
        val_step = 20

    with wandb.init(project="moving_part_segmentation", config=args):
        wandb.watch(model, log="all", log_freq=10)
        for epoch in range(args.resume_epoch,args.epochs):
            is_best=False
            adjust_learning_rate(optimizer,epoch, args)

            losses, errors= run_all(run_type="train",additional_arguments=additional_arguments,data_loader=train_loader, model=model,multitaskloss_instance=multitaskloss_instance,optimizer=optimizer,
                                                        motion_lfn=motion_lfn,orientation_lfn=orientation_lfn,residual_lfn=residual_lfn,point_lfn=point_lfn,vis=False)
            print_le(writer,epoch,optimizer,losses=losses,errors=errors,split="train",best_epoch=best_epoch)
            is_best_train = losses[0] < best_train_loss
            if(is_best_train):
                best_train_loss = losses[0]
                val_losses,val_errors = run_all(run_type="val",additional_arguments=additional_arguments,data_loader=val_loader, model=model,multitaskloss_instance=multitaskloss_instance,optimizer=optimizer,
                                                        motion_lfn=motion_lfn,orientation_lfn=orientation_lfn,residual_lfn=residual_lfn,point_lfn=point_lfn,vis=False)
                print_le(writer,epoch,optimizer,losses=val_losses,errors=val_errors,split="val",best_epoch=best_epoch)
                is_best = val_losses[0] < best_loss
            if is_best:
                best_loss = val_losses[0]
                best_epoch = epoch
                test_losses,test_errors = run_all(run_type="test",additional_arguments=additional_arguments,data_loader=test_loader, model=model,multitaskloss_instance=multitaskloss_instance,optimizer=optimizer,
                                                        motion_lfn=motion_lfn,orientation_lfn=orientation_lfn,residual_lfn=residual_lfn,point_lfn=point_lfn,vis=False)
                print_le(writer,epoch,optimizer,losses=test_losses,errors=test_errors,split="test",best_epoch=best_epoch)
            if args.use_multi_loss:
                save_checkpoint(state={"epoch": epoch, "state_dict": {"state_dict":model.state_dict(),"multi_loss_dict":multitaskloss_instance.state_dict()}, "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                                        is_best=is_best, checkpoint_folder=checkpoint)  
            else:
                save_checkpoint(state={"epoch": epoch, "state_dict": {"state_dict":model.state_dict(),"multi_loss_dict":None}, "best_loss": best_loss, "optimizer": optimizer.state_dict()},
                                        is_best=is_best, checkpoint_folder=checkpoint)
            
            

    loss,type_loss,orientation_loss,residual_loss,point_loss = test_losses
    type_accuracy,orientation_error, point_error = test_errors
    row= [loss,type_loss,orientation_loss,residual_loss,point_loss,type_accuracy,orientation_error, point_error]
    
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(row)

    # print('running the final train dataset in eval mode')
    # train_losses,train_errors = run_all(run_type="test",additional_arguments=additional_arguments,data_loader=train_loader, model=model,multitaskloss_instance=multitaskloss_instance,optimizer=optimizer,
    #                                                     motion_lfn=motion_lfn,orientation_lfn=orientation_lfn,residual_lfn=residual_lfn,point_lfn=point_lfn,vis=False)
    # print_le(writer,epoch,optimizer,losses=train_losses,errors=train_errors,split="train",best_epoch=best_epoch)
    # for visualizing the results afterwards
    best_epoch,best_loss = load_models(args=args,checkpoint=checkpoint,model=model,optimizer=optimizer,
                                           multitaskloss_instance=multitaskloss_instance)
    test_losses,test_errors = run_all(run_type="test",additional_arguments=additional_arguments,data_loader=test_loader, model=model,multitaskloss_instance=multitaskloss_instance,optimizer=optimizer,
                                                        motion_lfn=motion_lfn,orientation_lfn=orientation_lfn,residual_lfn=residual_lfn,point_lfn=point_lfn,vis=True)
    print_le(writer,epoch,optimizer,losses=test_losses,errors=test_errors,split="test",best_epoch=best_epoch)
    writer.flush()
    writer.close()
    print('finished training')
    
        



if __name__ == "__main__":
    args = get_args()
    # categories = ["Bottle", "Refrigerator", "Display", "Laptop", "Knife", "Clock",  "Scissors", "Door", "Pen", "Pliers", "Oven", "Cart", "USB"]
    categories = ["Display", "Knife", "Door", "Pen", "Pliers", "Oven", "Cart", "USB", "all"]
    args.category = categories[args.category_index]
    if args.use_seed:
        print("seeding")
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        np.random.seed(0)
    main(args)
   


# # validation function
# def val_legacy(additional_arguments,val_loader, model, multitaskloss_instance,
#         motion_lfn,orientation_lfn,residual_lfn,point_lfn,vis=False):

#     args = additional_arguments['args']
#     loss_weights = additional_arguments['loss_weights']
    
#     model.eval()
#     if args.use_multi_loss:
#         multitaskloss_instance.eval()

#     loss_sum = 0
#     count = 0

#     type_accuracy_sum = []
#     orientation_error_sum = []
#     point_error_sum = []
    

#     type_loss_sum =  0
#     orientation_loss_sum = 0
#     residual_loss_sum = 0
#     point_loss_sum = 0

#     oriented_axis = fibonacci_sphere()

#     for batch_data in tqdm(val_loader,disable=True):
#         with torch.no_grad():

#             output = model(batch_data)
            

#             (type_loss, orientation_loss, residual_loss, point_loss),(accuracy,orientation_error,point_error),(predicted_axis_mov,predicted_rotation_point_rot,_,paths) = compute_loss(args,output,motion_lfn,orientation_lfn,residual_lfn,point_lfn)
#             losses = torch.stack([type_loss,orientation_loss,residual_loss,point_loss])
#             if args.use_multi_loss:
#                 losses = torch.stack([type_loss,orientation_loss,residual_loss,point_loss])
#                 loss = multitaskloss_instance(losses)
#             else:
#                 loss = (losses*loss_weights).sum()


#             loss_sum += (loss.cpu().item())
#             # loss_sum += 0
#             type_loss_sum += (type_loss.cpu().item())
#             orientation_loss_sum += (orientation_loss.cpu().item())
#             residual_loss_sum += (residual_loss.cpu().item())
#             point_loss_sum += (point_loss.cpu().item())
#             count+=1

#             if not args.pretraining:
#                 type_accuracy_sum.extend(accuracy.cpu().detach().tolist())
#                 orientation_error_sum.extend(orientation_error.cpu().detach().tolist())
#                 point_error_sum.extend(point_error.cpu().detach().tolist())
            
#             if vis:
#                 mov_index = 0
#                 rot_index = 0
#                 for i, path in enumerate(paths):
#                     input_data = torch.load(path)
#                     for j in range(1,len(input_data)):
#                         if (input_data[j]['motion_type']==0 or input_data[j]['motion_type'] == 1):
#                             input_data[j]['predicted_axis'] = predicted_axis_mov[mov_index].cpu().detach()
#                             input_data[j]['axis_error'] = orientation_error[mov_index].cpu().detach()
#                             input_data[j]['predicted_point'] = predicted_rotation_point_rot[rot_index].cpu().detach()
#                             input_data[j]['point_error'] = point_error[rot_index].cpu().detach()
#                             rot_index+=1
#                             mov_index+=1
#                         elif (input_data[j]['motion_type']==2):
#                             input_data[j]['predicted_axis'] = predicted_axis_mov[mov_index].cpu().detach()
#                             input_data[j]['axis_error'] = orientation_error[mov_index].cpu().detach()
#                             mov_index+=1
#                     torch.save(input_data,path)

#                 assert rot_index == predicted_rotation_point_rot.shape[0]
#                 assert mov_index == predicted_axis_mov.shape[0]
                           

                            






    
#         # ******************************************************************
#     return ((loss_sum / count, type_loss_sum/count, orientation_loss_sum/count, residual_loss_sum/count,point_loss_sum/count), 
#             (sum(type_accuracy_sum)/(len(type_accuracy_sum) if len(type_accuracy_sum) >0 else 1),
#              sum(orientation_error_sum)/(len(orientation_error_sum) if len(orientation_error_sum)>0 else 1),sum(point_error_sum)/(len(point_error_sum) if len(point_error_sum) >0 else 1)))
