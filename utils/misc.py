import sys,os,time
import torch
import math
import numpy as np
import json

def fibonacci_sphere():
    return torch.tensor([[1,0,0],[0,1,0],[0,0,1]])

def generate_pretraining_paths(args):
    local_time = time.localtime()
    time_string = time.strftime("%Y-%m-%d %H:%M", local_time)
    time_string = time_string.replace(" ", "_")
    time_string = time_string.replace(":", "_")
    base_output = os.path.join(args.base_output,args.category,'Pretraining',time_string)
    checkpoint = os.path.join(base_output,args.checkpoint)
    runs = os.path.join(base_output,args.runs)
    for path in [checkpoint,runs]:
        os.makedirs(path,exist_ok=True)
    return checkpoint,runs,None


def generate_fine_tuning_paths(args):
    base_output = os.path.join(args.base_output,args.category,'Pretraining')
    if os.path.exists(base_output) and len(os.listdir(base_output)):
        time_stamps = os.listdir(base_output)
        time_stamps.sort(reverse=True)
        latest_time_stamp = time_stamps[0]
    else:
        if args.split_index == 0:
            print("pretraining is empty and the training index is 0 creating a new time stamp")
            local_time = time.localtime()
            time_string = time.strftime("%Y-%m-%d %H:%M", local_time)
            time_string = time_string.replace(" ", "_")
            time_string = time_string.replace(":", "_")
            latest_time_stamp = time_string
        else:
            print("pretraining is empty but the training index is not 0 using a prev time stamp")
            base_output = os.path.join(args.base_output,args.category,'fine_tuning')
            time_stamps = os.listdir(base_output)
            time_stamps.sort(reverse=True)
            latest_time_stamp = time_stamps[0]
            assert int(latest_time_stamp)<3


    base_output = os.path.join(args.base_output,args.category,'fine_tuning',latest_time_stamp,str(args.split_index))
    checkpoint = os.path.join(base_output,args.checkpoint)
    runs = os.path.join(base_output,args.runs)
    csv = os.path.join(base_output,f'metrics.csv')
    for path in [checkpoint,runs]:
        os.makedirs(path,exist_ok=True)
    return checkpoint,runs,csv



def retrieve_paths(args):
    assert args.resume_train or args.test
    base_output = os.path.join(args.base_output,args.category,"Pretraining" if args.pretraining else "fine_tuning")
    time_stamps = os.listdir(base_output)
    time_stamps.sort(reverse=True)
    latest_time_stamp = time_stamps[0]
    base_output = os.path.join(base_output,latest_time_stamp)
    if not args.pretraining:
        base_output = os.path.join(base_output,str(args.split_index))


    checkpoint = os.path.join(base_output,args.checkpoint)
    runs = os.path.join(base_output,args.runs)
    csv = os.path.join(base_output,'metrics.csv')
    return checkpoint, runs,csv

def retrive_checkpoint_from_directory(args):
    base_output = args.test_ckpt_dir
    checkpoint = os.path.join(base_output,args.best_model)
    runs = os.path.join(base_output,args.runs)
    logs = os.path.join(base_output,args.logs)
    return checkpoint,runs,logs


def state_dict(checkpoint):
    all_states = torch.load(os.path.join(checkpoint,'model_best.pth.tar'))
    return all_states['state_dict'], all_states['optimizer'],all_states['epoch'],all_states['best_loss'] 


def redirect_output_to_file(output_file_path):
    """Redirect stdout and stderr to a specified log file and return the originals and file handle."""
    original_stdout = sys.stdout  # Save the current stdout
    original_stderr = sys.stderr  # Save the current stderr

    log_file = open(output_file_path, 'a')  # Open the specified log file in append mode
    sys.stdout = log_file
    sys.stderr = log_file

    return original_stdout, original_stderr, log_file

def restore_original_output(original_stdout, original_stderr, log_file):
    """Restore stdout and stderr to their original settings and close the log file."""
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()



def loss_funs(args,motion_weights,orientation_type_weights):
    motion_lfn = torch.nn.CrossEntropyLoss(weight=motion_weights)
    #orientation_lfn = torch.nn.CrossEntropyLoss(weight=orientation_type_weights)
    orientation_lfn = torch.nn.BCEWithLogitsLoss()
    residual_lfn = torch.nn.MSELoss()
    point_lfn = torch.nn.MSELoss()
    #point_lfn = point_loss_function
    return motion_lfn,orientation_lfn,residual_lfn,point_lfn


def multilabel_accuracy(y_pred, y_true, threshold=0.5):
    y_pred = torch.argmax(y_pred,dim=1,keepdim=True)
    y_true = torch.argmax(y_true,dim=1,keepdim=True)
    correct_predictions = (y_pred == y_true).float()
    return correct_predictions.mean()





def compute_relevant(args,output):
    discrete_axis = fibonacci_sphere()
    discrete_axis = discrete_axis.to(output['predicted_motion_type'].device)
    all_predicted_motion_type = output['predicted_motion_type']
    all_predicted_orientation_label = output['predicted_orientation_label']
    all_predicted_residual = output['predicted_residual']
    all_predicted_rotation_point = output['predicted_rotation_point']
    all_g_motion_type = output['g_motion_type']
    all_g_orientation_label = output['g_orientation_label']
    all_g_residual = output['g_residual']
    all_g_rotation_point = output['g_rotation_point']
    all_g_axis = output['g_truth_axis']

    
    mask_mov = all_g_motion_type[:, -1] != 1
    mask_rot = (all_g_motion_type[:, 0] == 1) | (all_g_motion_type[:, 1] ==1)

    if not args.pretraining:
        movable_ids_orientation = (output['movable_ids'][mask_mov]>=1000).all()
        movabale_ids_point = (output['movable_ids'][mask_rot]>=1000).all()
        assert movable_ids_orientation.item() and movabale_ids_point.item() ## all the movabale ids from the orinetation mask and point belong to a movable part
    

   




    # this takes of getting the right parts
    
    predicted_orientation_label_mov = all_predicted_orientation_label[mask_mov]
    g_orientation_label_mov = all_g_orientation_label[mask_mov]
    predicted_residual_mov = all_predicted_residual[mask_mov]
    g_residual_mov = all_g_residual[mask_mov]
    g_axis_mov = all_g_axis[mask_mov]
    

    predicted_rotation_point_rot = all_predicted_rotation_point[mask_rot]
    g_rotation_point_rot = all_g_rotation_point[mask_rot]

    predicted_orientation_label_rot = all_predicted_orientation_label[mask_rot]
    g_orientation_label_rot = all_g_orientation_label[mask_rot] # only parts that have that the rotation point
    predicted_residual_rot = all_predicted_residual[mask_rot]
    g_residual_rot = all_g_residual[mask_rot]
    g_axis_rot = all_g_axis[mask_rot]
    
    # getting the predicted axis mov and the ground truth axis mov for the movable part
    if not args.pretraining:
        max_predicted_label_mov = torch.argmax(predicted_orientation_label_mov,dim=1)
        predicted_discrete_axis_mov = discrete_axis[max_predicted_label_mov]
        max_predicted_residual_mov = predicted_residual_mov[torch.arange(max_predicted_label_mov.shape[0]),max_predicted_label_mov]
        predicted_axis_mov = predicted_discrete_axis_mov + max_predicted_residual_mov
       

    if not args.pretraining:
        max_predicted_label_rot = torch.argmax(predicted_orientation_label_rot,dim=1)
        predicted_discrete_axis_rot = discrete_axis[max_predicted_label_rot]
        max_predicted_residual_rot = predicted_residual_rot[torch.arange(max_predicted_label_rot.shape[0]),max_predicted_label_rot]
        predicted_axis_rot = predicted_discrete_axis_rot + max_predicted_residual_rot
        max_predicted_rotation_point_rot = predicted_rotation_point_rot[torch.arange(max_predicted_label_rot.shape[0]),max_predicted_label_rot]





    masking_orientations_mov = g_orientation_label_mov.reshape(-1)
    masking_orientations_mov = masking_orientations_mov==1
    predicted_residual_mov = predicted_residual_mov.reshape(-1,3)[masking_orientations_mov]
    g_residual_mov = g_residual_mov.reshape(-1,3)[masking_orientations_mov]
    g_axis_mov = g_axis_mov.reshape(-1,3)[masking_orientations_mov]

    
   


   
    
    masking_orientations_rot = g_orientation_label_rot.reshape(-1)
    masking_orientations_rot = masking_orientations_rot==1
    predicted_rotation_point_rot = predicted_rotation_point_rot.reshape(-1,3)[masking_orientations_rot]
    g_rotation_point_rot = g_rotation_point_rot.reshape(-1,3)[masking_orientations_rot]
    g_axis_rot = g_axis_rot.reshape(-1,3)[masking_orientations_rot]
  

    



    results = {'all_predicted_motion_type':all_predicted_motion_type,
                'all_g_motion_type':all_g_motion_type,
                'predicted_orientation_label_mov':predicted_orientation_label_mov,
                'g_orientation_label_mov':g_orientation_label_mov,
                'predicted_residual_mov':predicted_residual_mov,
                'g_residual_mov':g_residual_mov,
                'predicted_rotation_point_rot':predicted_rotation_point_rot,
                'g_rotation_point_rot':g_rotation_point_rot,
                'g_axis_mov':g_axis_mov,
                'predicted_axis_mov':predicted_axis_mov if not args.pretraining else None,
                'max_predicted_rotation_point_rot': max_predicted_rotation_point_rot if not args.pretraining else None,
                'g_axis_rot': g_axis_rot
                }

    return results





def compute_loss(args,output,motion_lfn,orientation_lfn,residual_lfn,point_lfn):
    results = compute_relevant(args,output)
    all_predicted_motion_type = results['all_predicted_motion_type']
    all_g_motion_type = results['all_g_motion_type']
    predicted_orientation_label_mov = results['predicted_orientation_label_mov']
    g_orientation_label_mov = results['g_orientation_label_mov']
    predicted_residual_mov = results['predicted_residual_mov']
    g_residual_mov = results['g_residual_mov']
    predicted_rotation_point_rot = results['predicted_rotation_point_rot']
    g_rotation_point_rot = results['g_rotation_point_rot']
    g_axis_mov = results['g_axis_mov']
    predicted_axis_mov = results['predicted_axis_mov']
    max_predicted_rotation_point_rot = results['max_predicted_rotation_point_rot']
    g_axis_rot = results['g_axis_rot']
    paths = output['paths']
    num_parts = output['num_parts']



    # print(f'{predicted_orientation_label.shape=}')
    # print(f'{g_orientation_label.shape=}')
    # print(f'{g_residual.shape=}')
    # print(f'{predicted_residual.shape=}')
    # print(f'{predicted_rotation_point.shape=}')
    # print(f'{g_rotation_point.shape=}')
    # raise KeyboardInterrupt





    # this remains same during
    type_loss = motion_lfn(all_predicted_motion_type,all_g_motion_type)
    if (g_orientation_label_mov.shape[0]!=0):
        orientation_loss = orientation_lfn(predicted_orientation_label_mov,g_orientation_label_mov)
        residual_loss = residual_lfn(predicted_residual_mov,g_residual_mov)
        if not args.pretraining:
            orientation_error = calculate_axis_error(predicted_axis_mov,g_axis_mov)
        else:
            orientation_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)

    else:
        orientation_loss = torch.tensor(0,dtype=torch.float,device=all_predicted_motion_type.device)
        residual_loss = torch.tensor(0,dtype=torch.float,device=all_predicted_motion_type.device)
        orientation_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)


  
    if (g_rotation_point_rot.shape[0]!=0):
        #point_loss  = point_lfn(predicted_rotation_point_rot,g_rotation_point_rot,g_axis_rot)
        point_loss  = point_lfn(predicted_rotation_point_rot,g_rotation_point_rot)
        if not args.pretraining:
            point_error = calculate_point_error(max_predicted_rotation_point_rot,g_rotation_point_rot,g_axis_rot)
        else:
            point_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)


    else:
        point_loss = torch.tensor(0,dtype=torch.float,device=all_predicted_motion_type.device)
        point_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)


    return (type_loss, orientation_loss, residual_loss, point_loss),(orientation_error,point_error),(predicted_axis_mov,predicted_rotation_point_rot,num_parts,paths)


def point_loss_function(predicted_rotation_point_rot,g_rotation_point_rot,g_axis_rot):
    cross_product = torch.cross(g_axis_rot,g_rotation_point_rot-predicted_rotation_point_rot,dim=1)
    distance = torch.linalg.norm(cross_product,dim=1)/torch.linalg.norm(g_axis_rot,dim=1)
    sqaured = distance**2
    return torch.mean(sqaured)


@torch.no_grad()
def calculate_axis_error(p_drt, t_drt):
    p_drt_n = p_drt/torch.linalg.norm(p_drt,dim=1,keepdim=True)
    t_drt_n = t_drt/torch.linalg.norm(t_drt,dim=1,keepdim=True)
    drt_cos = torch.sum(p_drt_n * t_drt_n, dim=1) / (torch.norm(p_drt_n, dim=1) * torch.norm(t_drt_n, dim=1))
    drt_error = torch.rad2deg(torch.acos(drt_cos))
    drt_error[drt_error > 90] = 180 - drt_error[drt_error > 90]
    return drt_error

@torch.no_grad()
def calculate_point_error(p_pos, t_pos, t_drt):
    t_drt_n = t_drt/torch.linalg.norm(t_drt,dim=1,keepdim=True)
    cross_product = torch.cross(p_pos - t_pos, t_drt_n,dim=1)
    pos_error = torch.norm(cross_product,dim=1) / torch.norm(t_drt_n, dim=1)
    return pos_error

@torch.no_grad()
def compute_errors(args,oriented_axis,output):
    return
    
    # result_dict = compute_relevant(args,output)
    # predicted_motion_type,g_motion_type,predicted_orientation_label,g_orientation_label,predicted_residual,g_residual,predicted_rotation_point,g_rotation_point = **results
    

    # # to compute the type error
    # y_pred = torch.softmax(predicted_motion_type,dim=-1)
    # y_true = g_motion_type
    # motion_type_accuracy = multilabel_accuracy(y_pred,y_true)


    # predicted_directions = oriented_axis + output['predicted_residual']
    # predicted_directions = predicted_directions/torch.linalg.norm(predicted_directions,dim=-1,keepdim=True)
    # truth_directions = output['g_truth_axis']

    # predicted_position = output['predicted_rotation_point']
    # truth_position = output['g_rotation_point']


    # # masks
    # orientation_mask = g_motion_type[:, -1] != 1
    # point_mask = (g_motion_type[:, 0] == 1) | (g_motion_type[:, 1] ==1)
    # if not args.pretraining
    #     movable_ids_orientation = (output['movable_ids'][orientation_mask]>=1000).all()
    #     movabale_ids_point = (output['movable_ids'][point_mask]>=1000).all()
    #     assert movable_ids_orientation.item() and movabale_ids_point.item() ## all the movabale ids from the orinetation mask and point belong to a movable part
    

    


    # p_drt = predicted_directions[orientation_mask]
    # t_drt = truth_directions[orientation_mask]


    # assert (orientation_mask==output['movable']).all()


    # # print(output['paths'])
    # # print(f'{p_drt=}')
    # # print(f'{t_drt=}')


    # if p_drt.shape[0]!=0:
    #     orientation_error = calculate_axis_error(p_drt,t_drt)
    # else:
    #     orientation_error  = torch.tensor([],device='cuda:0')


    # p_pos = predicted_position[point_mask]
    # t_pos = truth_position[point_mask]
    # t_drt = truth_directions[point_mask]

    # if p_pos.shape[0]!=0:
    #     position_error = calculate_point_error(p_pos,t_pos,t_drt)
    # else:
    #     position_error = torch.tensor([],device='cuda:0')
    
    # return motion_type_accuracy,orientation_error,position_error



def axis_points(oriented_axis,output):

    axis_labels = torch.argmax(output['predicted_orientation_label'],dim=1)
    predicted_directions = oriented_axis[axis_labels]+ output['predicted_residual'][torch.arange(output['predicted_residual'].shape[0]),axis_labels,:]
    predicted_directions = predicted_directions/torch.linalg.norm(predicted_directions,dim=1,keepdim=True)

    predicted_positions = output['predicted_rotation_point']
    return predicted_directions,predicted_positions
    


    

def compute_stats(args,split='train'):
    with open (os.path.join(args.base_dir, args.data_root,f'{split}.json'),'r') as file:
        data = json.load(file)
    
    all_paths = []
    if args.category != 'all':
        category_paths = data[args.category]
        for path in category_paths:
            shape, part = path.split("|")
            all_paths.append(os.path.join(args.base_dir,args.data_root,shape,part))
    else:
        for category in data:
            category_paths = data[category]
            for path in category_paths:
                shape, part = path.split("|")
                all_paths.append(os.path.join(args.base_dir,args.data_root,shape,part))



    label_bin = [0 for _ in range (3)]
    type_bin = [0 for _ in range(4)]

    for path in all_paths:
        part_dict = torch.load(path)
        if (part_dict['is_cls']):
            continue
        motion_type = part_dict['motion_type']
        if type(motion_type) ==list:
            motion_type = motion_type[0]
        type_bin[motion_type]+=1
        if motion_type != 3:
            label_bin[part_dict['label']]+=1

    return type_bin,label_bin


def compute_stats_transform(args,split='train',split_index=0):
    with open (os.path.join(args.base_dir, args.data_root,f'{split}_{int(split_index)}.json'),'r') as file:
        data = json.load(file)
    
    all_paths = []
    if args.category != 'all':
        category_paths = data[args.category]
        for path in category_paths:
            all_paths.append(os.path.join(args.base_dir,args.data_root,path))
    else:
        for category in data:
            category_paths = data[category]
            for path in category_paths:
                all_paths.append(os.path.join(args.base_dir,args.data_root,path))


    label_bin = [0 for _ in range (3)]
    type_bin = [0 for _ in range(4)]
    num_parts = [0 for _ in range(250)]
    #all_moving_shape_parts = [0 for _ in range(250)]

    for path in all_paths:
        part_list = torch.load(path)
        num_parts[len(part_list)]+=1
        for part_dict in part_list:
            if (part_dict['is_cls']==False):
                motion_type = part_dict['motion_type']
                if type(motion_type) ==list:
                    motion_type = motion_type[0]
                type_bin[motion_type]+=1
                if motion_type != 3:
                    labels = part_dict['label']
                    for label in labels:
                        label_bin[label]+=1
        del part_list

    
    
    return type_bin,label_bin,num_parts


def print_le(writer,epoch,optimizer,losses,errors,split,best_epoch=-1):
    loss,type_loss,orientation_loss,residual_loss,point_loss = losses
    type_accuracy,orientation_error, point_error = errors

    print(f"Epoch {epoch:d}. {split}_loss: {loss:.8f}. type_loss: {type_loss:.8f}. orientation_loss: {orientation_loss:.8f}. residual_loss: {residual_loss:.8f}. point_loss: {point_loss:.8f}. accuracy {type_accuracy}. orientation_error {orientation_error}. point_error {point_error}. Current lr : {optimizer.param_groups[0]['lr']:.8f}. best_epoch: {best_epoch}")
    writer.add_scalar(f"Loss/{split}", loss, epoch)
    writer.add_scalar(f"type_loss/{split}", type_loss, epoch)
    writer.add_scalar(f"orientation_loss/{split}", orientation_loss, epoch)
    writer.add_scalar(f"residual_loss/{split}", residual_loss, epoch)
    writer.add_scalar(f"point_loss/{split}", point_loss, epoch)

    writer.add_scalar(f"type_accuracy/{split}", type_accuracy, epoch)
    writer.add_scalar(f"orientation_error/{split}", orientation_error, epoch)
    writer.add_scalar(f"point_error/{split}", point_error, epoch)

    # if val_errors != None:
    #     loss,type_loss,orientation_loss,residual_loss,point_loss = val_losses
    #     type_accuracy,orientation_error, point_error = val_errors
    #     if (is_test==False):
    #         print(f"Epoch {epoch:d}. val_loss: {loss:.8f}. type_loss: {type_loss:.8f}. orientation_loss: {orientation_loss:.8f}. residual_loss: {residual_loss:.8f}. point_loss: {point_loss:.8f}. accuracy {type_accuracy}. best_epoch: {best_epoch} orientation_error {orientation_error}. point_error {point_error}. Current lr : {optimizer.param_groups[0]['lr']:.8f}.")
    #         writer.add_scalar("Loss/val", loss, epoch)
    #         writer.add_scalar("type_loss/val", type_loss, epoch)
    #         writer.add_scalar("orientation_loss/val", orientation_loss, epoch)
    #         writer.add_scalar("residual_loss/val", residual_loss, epoch)
    #         writer.add_scalar("point_loss/val", point_loss, epoch)

    #         writer.add_scalar("type_accuracy/val", type_accuracy, epoch)
    #         writer.add_scalar("orientation_error/val", orientation_error, epoch)
    #         writer.add_scalar("point_error/val", point_error, epoch)
    #     else:
    #         print(f"Epoch {epoch:d}. test_loss: {loss:.8f}. type_loss: {type_loss:.8f}. orientation_loss: {orientation_loss:.8f}. residual_loss: {residual_loss:.8f}. point_loss: {point_loss:.8f}. accuracy {type_accuracy}. orientation_error {orientation_error}. point_error {point_error}. Current lr : {optimizer.param_groups[0]['lr']:.8f}.")

    
        


    # can remove this, helpful for debugging
    # this tells which part belongs to which index
    # paths = output['paths']
    # paths_index = torch.arange(len(output['paths']),dtype=torch.int).to(y_pred.device)
    # repeat_index = output['num_parts'].to(y_pred.device)-1
    # paths_index = paths_index.repeat_interleave(repeat_index)

    #axis_labels = torch.argmax(output['predicted_orientation_label'],dim=1)
    #predicted_directions = oriented_axis + output['predicted_residual'][torch.arange(output['predicted_residual'].shape[0]),axis_labels,:]




#  # for all parts predicted axis start from here
#     if not args.pretraining:
#         max_predicted_label_all = torch.argmax(all_predicted_orientation_label,dim=1)
#         predicted_discrete_axis_all = discrete_axis[max_predicted_label_all]
#         max_predicted_residual_all = all_predicted_residual[torch.arange(max_predicted_label_all.shape[0]),max_predicted_label_all]
#         predicted_axis_all = predicted_discrete_axis_all + max_predicted_residual_all
#         g_axis_all = all_g_axis
#         max_predicted_rotation_point_all = all_predicted_rotation_point[torch.arange(max_predicted_label_all.shape[0]),max_predicted_label_all]
#         print(f'{predicted_axis_all.shape=}')
#         print(f'{g_axis_all.shape=}')
#         print(f'{max_predicted_rotation_point_all.shape=}')


# def point_loss_function(predicted_rotation_point_rot, g_rotation_point_rot, g_axis_rot):
#     # Clone the tensors if they are going to be modified
#     g_axis_rot = g_axis_rot.clone()
#     g_rotation_point_rot = g_rotation_point_rot.clone()
#     predicted_rotation_point_rot = predicted_rotation_point_rot.clone()

#     cross_product = torch.cross(g_axis_rot, g_rotation_point_rot - predicted_rotation_point_rot, dim=1)
#     distance = torch.linalg.norm(cross_product, dim=1) / torch.linalg.norm(g_axis_rot, dim=1)
#     return torch.mean(distance**2)