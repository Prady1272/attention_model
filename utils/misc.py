import sys,os,time
import torch
import math
import numpy as np
import json
from PIL import Image

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
    pretrain_exists = False
    pretrain_checkpoint = None
    pretrain_base_checkpoint = None
    if args.use_pretraining:
        base_output = os.path.join(args.base_output,args.category,'Pretraining')
        assert os.path.exists(base_output) and len(os.listdir(base_output))
        print(f'using the pretrain path for starting new fine tuning')
        pretrain_exists = True
        time_stamps = os.listdir(base_output)
        time_stamps.sort(reverse=True)
        latest_time_stamp = time_stamps[0]
        pretrain_checkpoint = os.path.join(base_output,latest_time_stamp,'checkpoint')

    if args.use_pretraining:
        base_output = os.path.join(args.base_output,args.category,'fine_tuning',latest_time_stamp,str(args.split_index))
    else:
        local_time = time.localtime()
        time_string = time.strftime("%Y-%m-%d %H:%M", local_time)
        time_string = time_string.replace(" ", "_")
        time_string = time_string.replace(":", "_")
        print('making a new path. so no pretrained for fine tuning')
        base_output = os.path.join(args.base_output,args.category,'fine_tuning',time_string,str(args.split_index))

    checkpoint = os.path.join(base_output,args.checkpoint)
    runs = os.path.join(base_output,args.runs)
    csv = os.path.join(base_output,f'metrics.csv')
    for path in [checkpoint,runs]:
        os.makedirs(path,exist_ok=True)
    return pretrain_checkpoint,checkpoint,runs,csv



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


def state_dict(args,checkpoint,latest):
    if latest:
        print(f'Returning the LATEST pth.tar from {checkpoint=}')
        all_states = torch.load(os.path.join(checkpoint,"checkpoint.pth.tar"))
    else:
        print(f'Returning the BEST model from {checkpoint=}')
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
    print(f'{motion_weights=}')
    print(f'{orientation_type_weights=}')
    motion_lfn = torch.nn.CrossEntropyLoss(weight=motion_weights,reduction=args.loss_reduction)
    #orientation_lfn = torch.nn.CrossEntropyLoss(weight=orientation_type_weights)
    orientation_lfn = torch.nn.BCEWithLogitsLoss(reduction=args.loss_reduction,pos_weight=orientation_type_weights)
    residual_lfn = torch.nn.L1Loss(reduction=args.loss_reduction)
    point_lfn = torch.nn.L1Loss(reduction=args.loss_reduction)
    #point_lfn = point_loss_function
    return motion_lfn,orientation_lfn,residual_lfn,point_lfn


@torch.no_grad()
def multilabel_accuracy(y_pred, y_true, threshold=0.5):
    # print(f'{y_true=}')
    y_pred = torch.argmax(y_pred,dim=1)
    y_true = torch.argmax(y_true,dim=1)
    correct_predictions = (y_pred == y_true).float()
    
    # print(f'{y_pred=}')
    # print(f'{correct_predictions=}')
    rotationary_index = y_true==0
    # print(f'{rotationary_index=}')
    correct_rotationary = correct_predictions[rotationary_index]
    # print(f'{correct_rotationary=}')

    screw_index = y_true==1 
    # print(f'{screw_index=}')
    correct_screw = correct_predictions[screw_index]
    # print(f'{correct_screw=}')

    trans_index = y_true==2
    # print(f'{trans_index=}')
    correct_trans = correct_predictions[trans_index]
    # print(f'{correct_trans=}')

    fixed_index = y_true==3
    # print(f'{fixed_index=}')
    correct_fixed = correct_predictions[fixed_index]
    # print(f'{correct_fixed=}')

    return (correct_predictions,correct_rotationary,correct_screw,correct_trans,correct_fixed),y_pred





def compute_relevant(args,output,pretraining):
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
    all_centroid = output['centroid']
    all_m = output["m"]
    all_part_index = output['part_index']

    # print("starting new")
    # print(f'{all_g_motion_type=}')
    # print(f'{all_predicted_motion_type=}')
    # print(f'{all_g_axis=}')
    # print(f'{all_g_orientation_label=}')
    # print(f'{all_predicted_orientation_label=}')
    # print(f'{all_g_residual=}')
    # print(f'{all_predicted_residual=}')
    # print(f'{all_g_rotation_point=}')
    # print(f'{all_predicted_rotation_point=}')

    
    mask_mov = all_g_motion_type[:, -1] != 1
    # print(f'{mask_mov=}')
    mask_rot = (all_g_motion_type[:, 0] == 1) | (all_g_motion_type[:, 1] ==1)
    # print(f'{mask_rot=}')

    if not pretraining:
        movable_ids_orientation = (output['movable_ids'][mask_mov]>=1000).all()
        movabale_ids_point = (output['movable_ids'][mask_rot]>=1000).all()
        assert movable_ids_orientation.item() and movabale_ids_point.item() ## all the movabale ids from the orinetation mask and point belong to a movable part
    

   




    # this takes of getting the right parts
    
    predicted_orientation_label_mov = all_predicted_orientation_label[mask_mov]
    # print(f'{predicted_orientation_label_mov=}')
    g_orientation_label_mov = all_g_orientation_label[mask_mov]
    # print(f'{g_orientation_label_mov=}')
    predicted_residual_mov = all_predicted_residual[mask_mov]
    # print(f'{predicted_residual_mov=}')
    g_residual_mov = all_g_residual[mask_mov]
    # print(f'{g_residual_mov=}')
    
 
    g_axis_mov = all_g_axis[mask_mov]
    # print(f'{g_axis_mov=}')

    centroid_mov = all_centroid[mask_mov]

    part_index_mov = all_part_index[mask_mov]
    m_mov = all_m[mask_mov]
    

    predicted_rotation_point_rot = all_predicted_rotation_point[mask_rot]
    # print(f'{predicted_rotation_point_rot=}')
    g_rotation_point_rot = all_g_rotation_point[mask_rot]
    # print(f'{g_rotation_point_rot=}')

    predicted_orientation_label_rot = all_predicted_orientation_label[mask_rot]
    # print(f'{predicted_orientation_label_rot=}')
    g_orientation_label_rot = all_g_orientation_label[mask_rot] # only parts that have that the rotation point
    # print(f'{g_orientation_label_rot=}')
    predicted_residual_rot = all_predicted_residual[mask_rot]
    # print(f'{predicted_residual_rot=}')
    g_residual_rot = all_g_residual[mask_rot]
    # print(f'{g_residual_rot=}')
    g_axis_rot = all_g_axis[mask_rot]
    # print(f'{g_axis_rot=}')
    part_index_rot = all_part_index[mask_rot]
    
    # getting the predicted axis mov and the ground truth axis mov for the movable part
    if not pretraining:
        max_predicted_label_mov = torch.argmax(predicted_orientation_label_mov,dim=1)
        #print(f"{max_predicted_label_mov=}")
        predicted_discrete_axis_mov = discrete_axis[max_predicted_label_mov]
        #print(f"{predicted_discrete_axis_mov=}")
        max_predicted_residual_mov = predicted_residual_mov[torch.arange(max_predicted_label_mov.shape[0]),max_predicted_label_mov]
        #print(f"{max_predicted_residual_mov=}")
        predicted_axis_mov = predicted_discrete_axis_mov + max_predicted_residual_mov
        #print(f'{predicted_axis_mov=}')
       

    if not pretraining:
        max_predicted_label_rot = torch.argmax(predicted_orientation_label_rot,dim=1)
        predicted_discrete_axis_rot = discrete_axis[max_predicted_label_rot]
        max_predicted_residual_rot = predicted_residual_rot[torch.arange(max_predicted_label_rot.shape[0]),max_predicted_label_rot]
        predicted_axis_rot = predicted_discrete_axis_rot + max_predicted_residual_rot
        max_predicted_rotation_point_rot = predicted_rotation_point_rot[torch.arange(max_predicted_label_rot.shape[0]),max_predicted_label_rot]





    masking_orientations_mov = g_orientation_label_mov.reshape(-1)
    masking_orientations_mov = masking_orientations_mov==1
    # print(f'{masking_orientations_mov=}')
    predicted_residual_mov = predicted_residual_mov.reshape(-1,3)[masking_orientations_mov]
    # print(f'{predicted_residual_mov=}')
    g_residual_mov = g_residual_mov.reshape(-1,3)[masking_orientations_mov]
    # print(f'{g_residual_mov=}')
    g_axis_mov = g_axis_mov.reshape(-1,3)[masking_orientations_mov]
    # print(f'{g_axis_mov=}')

    
   


   
    
    masking_orientations_rot = g_orientation_label_rot.reshape(-1)
    masking_orientations_rot = masking_orientations_rot==1
    # print(f'{masking_orientations_rot=}')
    predicted_rotation_point_rot = predicted_rotation_point_rot.reshape(-1,3)[masking_orientations_rot]
    # print(f'{predicted_rotation_point_rot=}')
    g_rotation_point_rot = g_rotation_point_rot.reshape(-1,3)[masking_orientations_rot]
    # print(f'{g_rotation_point_rot=}')
    g_axis_rot = g_axis_rot.reshape(-1,3)[masking_orientations_rot]
    # print(f'{g_axis_rot=}')
  

    



    results = {'all_predicted_motion_type':all_predicted_motion_type,
                'all_g_motion_type':all_g_motion_type,
                'predicted_orientation_label_mov':predicted_orientation_label_mov,
                'g_orientation_label_mov':g_orientation_label_mov,
                'predicted_residual_mov':predicted_residual_mov,
                'g_residual_mov':g_residual_mov,
                'predicted_rotation_point_rot':predicted_rotation_point_rot,
                'g_rotation_point_rot':g_rotation_point_rot,
                'g_axis_mov':g_axis_mov,
                'predicted_axis_mov':predicted_axis_mov if not pretraining else None,
                'max_predicted_rotation_point_rot': max_predicted_rotation_point_rot if not pretraining else None,
                'g_axis_rot': g_axis_rot,
                "centroid_mov": centroid_mov,
                "m_mov": m_mov,
                'all_part_index': all_part_index,
                "part_index_mov":part_index_mov,
                "part_index_rot": part_index_rot,
                
                }

    return results





def compute_loss(args,output,motion_lfn,orientation_lfn,residual_lfn,point_lfn,pretraining):
    results = compute_relevant(args,output,pretraining=pretraining)
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
    centroid_mov = results['centroid_mov']
    m_mov = results['m_mov']
    all_part_index = results['all_part_index']
    part_index_mov = results['part_index_mov']
    part_index_rot = results['part_index_rot']
    
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
    # print("starting here")
    type_loss = motion_lfn(all_predicted_motion_type,all_g_motion_type)
    # if (torch.isnan(torch.tensor([type_loss])).any()):
    #     print(f'{output=}')
    #     print(f'{type_loss=}')
    #     print(f'{all_predicted_motion_type=}')
    #     print(f'{all_g_motion_type=}')
    accuracy,all_max_motion_predictions = multilabel_accuracy(y_pred=all_predicted_motion_type,y_true=all_g_motion_type)




    
    if (g_orientation_label_mov.shape[0]!=0):
        # print(f"{predicted_orientation_label_mov=}")
        # print(f"{g_orientation_label_mov=}")
        orientation_loss = orientation_lfn(predicted_orientation_label_mov,g_orientation_label_mov)
        # print(f"{orientation_loss=}")
        # print(f'{predicted_residual_mov=}')
        # print(f'{g_residual_mov=}')
        residual_loss = residual_lfn(predicted_residual_mov,g_residual_mov)
        #residual_loss = residual_loss + 0.1*torch.sum(predicted_residual_mov**2)
        # print(f'{residual_loss=}')
        if not pretraining:
            # print(f"{predicted_axis_mov=}")
            # print(f"{g_axis_mov=}")
            orientation_error = calculate_axis_error(predicted_axis_mov,g_axis_mov)
            # accuracy,all_max_motion_predictions = multilabel_accuracy(y_pred=all_predicted_motion_type,y_true=all_g_motion_type)
            # print(f'{orientation_error=}')
        else:
            # accuracy,all_max_motion_predictions = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device),None
            orientation_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)

    else:
        orientation_loss = torch.tensor(0,dtype=torch.float,device=all_predicted_motion_type.device)
        residual_loss = torch.tensor(0,dtype=torch.float,device=all_predicted_motion_type.device)
        orientation_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)



  
    if (g_rotation_point_rot.shape[0]!=0):
        #point_loss  = point_lfn(predicted_rotation_point_rot,g_rotation_point_rot,g_axis_rot)
        point_loss  = point_lfn(predicted_rotation_point_rot,g_rotation_point_rot)
        if not pretraining:
            point_error = calculate_point_error(max_predicted_rotation_point_rot,g_rotation_point_rot,g_axis_rot)
        else:
            point_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)


    else:
        point_loss = torch.tensor(0,dtype=torch.float,device=all_predicted_motion_type.device)
        point_error = torch.tensor([0],dtype=torch.float,device=all_predicted_motion_type.device)

    x,y,z = orientation_bins(predicted_orientation_label_mov=predicted_orientation_label_mov)
    return (type_loss, orientation_loss, residual_loss, point_loss),(accuracy,orientation_error,point_error,(x,y,z)),(predicted_axis_mov,predicted_rotation_point_rot, all_max_motion_predictions,predicted_orientation_label_mov, predicted_residual_mov,
                                                                                                              all_part_index,part_index_mov,part_index_rot,centroid_mov,m_mov,
                                                                                                              num_parts,paths)



@torch.no_grad()
def orientation_bins(predicted_orientation_label_mov):
    max_index = torch.argmax(predicted_orientation_label_mov,axis=1)
    x = torch.sum((max_index==0).float())
    y = torch.sum((max_index==1).float())
    z = torch.sum((max_index==2).float())
    return x,y,z





@torch.no_grad()
def vis_helper(args,details):
    (predicted_axis_mov,predicted_rotation_point_rot, all_max_motion_predictions,predicted_orientation_label_mov, predicted_residual_mov,
    all_part_index,part_index_mov,part_index_rot,centroid_mov,m_mov,
    num_parts,paths,
    orientation_error,point_error) = details
    suffix = f"{args.split_index}{'_ep' if args.encode_part else ''}{'_es'+str(args.num_layers) if args.encode_shape else ''}"
    print(f'{suffix=}')
    all_input_data = []
    num_mov  = []
    num_rot = []
    for i,path in enumerate(paths):
        input_data = torch.load(path,map_location='cpu')
        each_mov,each_rot = 0,0
        for j,data in enumerate(input_data):
            if data['motion_type'] != 3:
                each_mov+=1
            if data['motion_type'] == 0 or data['motion_type'] == 1:
                each_rot+=1
        num_mov.append(each_mov)
        num_rot.append(each_rot)

        assert num_parts[i] == len(input_data) # this is just checking if the num parts is right
        all_input_data.append(input_data)
    
   
    num_parts = num_parts-1
    # just checking if all the results of the right shapes
    assert torch.sum(num_parts) == all_max_motion_predictions.shape[0] and torch.sum(num_parts) == all_part_index.shape[0]
    assert sum(num_mov) == predicted_axis_mov.shape[0] and sum(num_mov) == predicted_orientation_label_mov.shape[0] and sum(num_mov) == centroid_mov.shape[0] and sum(num_mov) == m_mov.shape[0] and sum(num_mov) == part_index_mov.shape[0]
    assert sum(num_rot) ==  predicted_rotation_point_rot.shape[0] and sum(num_rot) == part_index_rot.shape[0]

    global_all_index = 0
    global_index_mov = 0
    global_index_rot = 0

    for i in range(len(all_input_data)):
        input_data = all_input_data[i]
        each_num_parts = num_parts[i]
        each_num_mov = num_mov[i]
        each_num_rot = num_rot[i]

        for j in range(each_num_parts):
            assert not input_data[all_part_index[global_all_index+j]]['is_cls'] # these assertion errors for just checking the type
            input_data[all_part_index[global_all_index+j]][f"predicted_motion_type_{suffix}"] = all_max_motion_predictions[global_all_index+j].cpu().detach()
        global_all_index+=each_num_parts

        for j in range(each_num_mov):
            assert input_data[part_index_mov[global_index_mov+j]]['motion_type'] !=3  # these assertion errors for just checking the type
            input_data[part_index_mov[global_index_mov+j]][f'predicted_axis_{suffix}'] = predicted_axis_mov[global_index_mov+j].cpu().detach()
            input_data[part_index_mov[global_index_mov+j]][f'axis_error_{suffix}'] = orientation_error[global_index_mov+j].cpu().detach()
            input_data[part_index_mov[global_index_mov+j]][f'predicted_orientation_label_mov_{suffix}'] = predicted_orientation_label_mov[global_index_mov+j].cpu().detach()
            input_data[part_index_mov[global_index_mov+j]][f'predicted_residual_mov_{suffix}'] = predicted_residual_mov[global_index_mov+j].cpu().detach()
            input_data[part_index_mov[global_index_mov+j]]['centroid'] = centroid_mov[global_index_mov+j].cpu().detach()
            input_data[part_index_mov[global_index_mov+j]]['m'] = m_mov[global_index_mov+j].cpu().detach()
        global_index_mov +=each_num_mov

        for j in range(each_num_rot):
            assert input_data[part_index_rot[global_index_rot+j]]['motion_type'] == 0 or input_data[part_index_rot[global_index_rot+j]]['motion_type'] == 1  # these assertion errors for just checking the type
            input_data[part_index_rot[global_index_rot+j]][f'predicted_point_{suffix}'] = predicted_rotation_point_rot[global_index_rot+j].cpu().detach()
            input_data[part_index_rot[global_index_rot+j]][f'point_error_{suffix}'] = point_error[global_index_rot+j].cpu().detach()

        global_index_rot +=each_num_rot
        torch.save(input_data,paths[i])



    # these assertion errors are for checking if we reach the final point
    assert global_all_index == all_max_motion_predictions.shape[0] and global_all_index == all_part_index.shape[0]
    assert global_index_mov == predicted_axis_mov.shape[0] and global_index_mov == part_index_mov.shape[0]
    assert global_index_rot == predicted_rotation_point_rot.shape[0] and global_index_rot == part_index_rot.shape[0]




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
    non_label_bins = [0 for _ in range(3)]
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
                    all_labels = [0,1,2]
                    for label in all_labels:
                        if label in labels:
                            label_bin[label]+=1
                        else:
                            non_label_bins[label]+=1
        del part_list

    
    
    return type_bin,label_bin,non_label_bins,num_parts


def print_le(writer,epoch,optimizer,losses,errors,split,best_epoch=-1):
    loss,type_loss,orientation_loss,residual_loss,point_loss = losses
    type_accuracy,type_rotation,type_screw,type_trans,type_fixed,x_bins,y_bins,z_bins, orientation_error, point_error = errors

    print(f"Epoch {epoch:d}. {split}_loss: {loss:.8f}. type_loss: {type_loss:.8f}. orientation_loss: {orientation_loss:.8f}. residual_loss: {residual_loss:.8f}. point_loss: {point_loss:.8f}. accuracy {type_accuracy:.8f}. rotation_accuracy {type_rotation:.8f}. screw_accuracy {type_screw:.8f}.trans_accuracy {type_trans:.8f}. fixed_accuracy {type_fixed:.8f}. orientation_error {orientation_error}. point_error {point_error}. x_bins {x_bins} y_bins {y_bins} z_bins {z_bins} Current lr : {optimizer.param_groups[0]['lr']:.8f}. best_epoch: {best_epoch}")
    writer.add_scalar(f"Loss/{split}", loss, epoch)
    writer.add_scalar(f"type_loss/{split}", type_loss, epoch)
    writer.add_scalar(f"orientation_loss/{split}", orientation_loss, epoch)
    writer.add_scalar(f"residual_loss/{split}", residual_loss, epoch)
    writer.add_scalar(f"point_loss/{split}", point_loss, epoch)

    writer.add_scalar(f"type_accuracy/{split}", type_accuracy, epoch)
    writer.add_scalar(f"type_rotation/{split}", type_rotation, epoch)
    writer.add_scalar(f"type_screw/{split}", type_screw, epoch)
    writer.add_scalar(f"type_trans/{split}", type_trans, epoch)
    writer.add_scalar(f"type_fixed/{split}", type_fixed, epoch)
    writer.add_scalar(f"orientation_error/{split}", orientation_error, epoch)
    writer.add_scalar(f"point_error/{split}", point_error, epoch)





@torch.no_grad()
def attn_weight_images(ouput,attn_output_weigths):
    
    part_index = ouput['part_index'].cpu().detach()
    indices = torch.nonzero(part_index == 1, as_tuple=False)
    index = indices[0,0]
    image = attn_output_weigths[index,0]
    image = image.cpu().detach().numpy()
    image = image.reshape(-1,32)
    print(f'{image.min()=}')
    print(f'{image.max()=}')
    image = image -  np.min(image)
    image /=image.max()
    image = Image.fromarray(np.uint8(image * 255) , 'L')



    return image












def final_metrics(metrics,count):
    loss_sum =              metrics["loss_sum"]
    type_accuracy_sum =     metrics["type_accuracy_sum"]
    type_rotationary_sum =  metrics["type_rotationary_sum"]
    type_screw_sum =        metrics["type_screw_sum"]
    type_trans_sum =        metrics["type_trans_sum"]
    type_fixed_sum =        metrics["type_fixed_sum"]
    x_bins =                metrics["x_bins"]
    y_bins =                metrics["y_bins"]
    z_bins =                metrics["z_bins"]
    orientation_error_sum = metrics["orientation_error_sum"]
    point_error_sum =       metrics["point_error_sum"]
    type_loss_sum =         metrics["type_loss_sum"]
    orientation_loss_sum =  metrics["orientation_loss_sum"]
    residual_loss_sum =     metrics['residual_loss_sum']
    point_loss_sum =        metrics["point_loss_sum"]

    return ((loss_sum / count, type_loss_sum/count, orientation_loss_sum/count, residual_loss_sum/count,point_loss_sum/count), 
            (sum(type_accuracy_sum)/(len(type_accuracy_sum) if len(type_accuracy_sum) >0 else 1),
             sum(type_rotationary_sum)/(len(type_rotationary_sum) if len(type_rotationary_sum) >0 else 1),
             sum(type_screw_sum)/(len(type_screw_sum) if len(type_screw_sum) >0 else 1),
             sum(type_trans_sum)/(len(type_trans_sum) if len(type_trans_sum) >0 else 1),
             sum(type_fixed_sum)/(len(type_fixed_sum) if len(type_fixed_sum) >0 else 1),
             x_bins,y_bins,z_bins,
             sum(orientation_error_sum)/(len(orientation_error_sum) if len(orientation_error_sum)>0 else 1),sum(point_error_sum)/(len(point_error_sum) if len(point_error_sum) >0 else 1)))








def pre_loaders(args,pretrain_dataset,preval_dataset,pretest_dataset,collate_fn):
    pretrain_loader = torch.utils.data.DataLoader(
                    pretrain_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True if args.num_workers != 0 else False,
                )
    
    pretest_loader = torch.utils.data.DataLoader(
                    pretest_dataset,
                    batch_size=args.batch_size*2,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True if args.num_workers != 0 else False,)
    

    preval_loader = torch.utils.data.DataLoader(
                    preval_dataset,
                    batch_size=args.batch_size*2,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=collate_fn,
                    persistent_workers=True if args.num_workers != 0 else False,)

    return pretrain_loader,pretest_loader,preval_loader








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