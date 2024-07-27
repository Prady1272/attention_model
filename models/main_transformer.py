import torch
import torch.nn as nn
from .Point_transformer_V3 import Point,PointTransformerV3
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
from .transformer_attention import Transformer
import torch_scatter
from torch import Tensor
import numpy as np
import math
from Pointnet_Pointnet2_pytorch.models.pointnet2_cls_ssg import get_model

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder  = Encode_Obb(input_size=3,channels=[256,512,512])
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        pass

    def forward(self, src: Tensor, src_key_padding_mask) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output
    





class MultiTaskLoss(torch.nn.Module):
  '''https://arxiv.org/abs/1705.07115'''
  def __init__(self, is_regression, reduction='none'):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses


class MLP(nn.Module):
    def __init__(self, input_size, channels, act_layer=nn.Tanh,last_layer_act=True,drop_prob=0.2):
        super(MLP,self).__init__()
        self.layers = nn.ModuleList()
        for size in channels:
            self.layers.append(nn.Linear(input_size, size))
            self.layers.append(nn.LayerNorm(size))
            self.layers.append(act_layer())
            self.layers.append(nn.Dropout(p=drop_prob))
            input_size = size  
            
        if not last_layer_act:
            self.layers = self.layers[:-3]
        self.initialize()        

    def initialize(self):
        for layer in self.layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data

# takes 2 tensor
class Encode_Obb(nn.Module):
    def __init__(self, input_size, channels, ):
        super(Encode_Obb,self).__init__()
        self.pp =  MLP(input_size=3,channels=channels)

    def forward(self, corners):
        corners_features = self.pp(corners)
        return torch.max(corners_features,dim=-2).values




class Predictor_Transformer(nn.Module):
    def __init__(
        self,
        device,
        reduce='mean',
        grid_size=0.05,
        backbone_out_channels=(32, 64, 128, 256, 512),
        max_sequence_length=50,
        num_heads = 32,
        num_layers = 1,
        cls_token = 'cls',
        enable_flash=True,

    ):
        super(Predictor_Transformer,self).__init__()
        self.device = device
        self.reduce = reduce
        self.grid_size = grid_size
        self.max_seq_len = max_sequence_length
        self.cls_token = cls_token
        self.input_self_attention_channel = backbone_out_channels[-1]

        # self.backbone = PointTransformerV3(enc_channels=backbone_out_channels,enable_flash = enable_flash)
        self.backbone = get_model(normal_channel=False)
        self.encode_obb  = Encode_Obb(input_size=3,channels=[256,512,512])
        #self.transformer =  Transformer(source_max_seq_len=max_sequence_length,embedding_dim=backbone_out_channels[-1],num_heads=num_heads,num_layers=num_layers)
        self.main_mlp = MLP(input_size=3*backbone_out_channels[-1],
                            channels=[int(3*backbone_out_channels[-1]),1024,1024,1024])
        
        self.motion_type_mlp = MLP(input_size=1024,
                            channels=[512,256,128,4],
                            last_layer_act=False)
        
        self.orientation_layer_mlp = MLP(input_size=1024,
                            channels=[512,256,128,16],
                            last_layer_act=False)
        
        self.residual_mlp = MLP(input_size=1024,
                            channels=[512,256,128,3],
                            last_layer_act=False)
        
        self.rotation_point_mlp = MLP(input_size=1024,
                            channels=[512,256,128,3],
                            last_layer_act=False)
        


        


    def forward(self, batch_dict):
        
        
        batch_size = batch_dict['num_parts'].shape[0]
        pc_size = 4096
        total_parts = torch.sum(batch_dict['num_parts'])
        part_coord = batch_dict['part_coord']
        part_coord = part_coord.reshape(batch_size,pc_size,3)
        part_features = self.backbone(part_coord)
        print(part_features.shape)
        return


        # part_offsets = torch.tensor([pc_size*(i+1) for i in range(total_parts)],dtype=torch.int)
        
        # part_input = dict( coord= part_coord.to(self.device),
        #                     feat= part_coord.to(self.device),
        #                     offset= part_offsets.to(self.device),
        #                     grid_size=self.grid_size)
        # part_out = self.backbone(part_input)
        # features = part_out.feat
        # offsets = part_out.offset
        # offsets = nn.functional.pad(offsets,(1,0))
        # part_features = torch_scatter.segment_csr(src=features,indptr=offsets,reduce=self.reduce)


        part_features = part_features.split(batch_dict['num_parts'].numpy().tolist())
        part_features = list(part_features)
        padded_part_features = pad_sequence(part_features, batch_first=True)
        max_length = padded_part_features.size(1)
        batch_size = padded_part_features.size(0)
        src_key_padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
        for i, seq in enumerate(part_features):
            src_key_padding_mask[i, len(seq):] = 0
            
        obb_corners = batch_dict['obb_corners']
        obb_features = self.encode_obb(obb_corners.to(self.device))

        # comibined_features = torch.cat([part_features,obb_features,shape_features],dim=1)
        # comibined_features = self.main_mlp(comibined_features)
        
        # predicted_motion_type = self.motion_type_mlp(comibined_features)
        # predicted_orientation_label = self.orientation_layer_mlp(comibined_features)
        # predicted_residual = self.residual_mlp(comibined_features)
        # predicted_rotation_point = self.residual_mlp(comibined_features)

        # output = dict(
        #     predicted_motion_type=predicted_motion_type,
        #     predicted_orientation_label=predicted_orientation_label,
        #     predicted_residual= predicted_residual,
        #     predicted_rotation_point= predicted_rotation_point,
        #     g_motion_type = batch_dict['motion_type'].to(self.device),
        #     g_orientation_label = batch_dict['orientation_label'].to(self.device),
        #     g_residual = batch_dict['residual'].to(self.device),
        #     g_rotation_point = batch_dict['rotation_point'].to(self.device),
        #     g_truth_axis = batch_dict['g_truth_axis'].to(self.device)
        # )
        # return output


        
        
       
        
        
        










# prev_1
#         # for data in batch_list:
#         #     print(data['offset'].shape)
#         #     print(data['coord'].shape)
#         #     print(data['vectors'][0].shape)
#         #     print(data['shape_id'])
#         #     print(data['component'])
#         cls_indexes = torch.tensor([data.index(self.cls_token) for data in batch['component']])
#         #print(f'{cls_indexes.shape=}')

     

#         point_transformer_input = dict(coord=batch['coord'].to(self.device),
#                                            feat=batch['coord'].to(self.device),
#                                            offset=batch['offset'].to(self.device),
#                                            grid_size=self.grid_size)
#         point_out = self.backbone(point_transformer_input)
#         features = point_out.feat


#         #setting things up
#         #getting partfeaturs
#         offsets = point_out.offset
#         offsets = nn.functional.pad(offsets,(1,0))
#         part_features = torch_scatter.segment_csr(src=features,indptr=offsets,reduce=self.reduce)
#         part_features = part_features.split(batch['seq_len'])

#         #getting concatenated_features
#         shape_features = [t[idx] for t, idx in zip(part_features, cls_indexes)]
#         component_features = [torch.cat([t[:idx],t[idx+1:]]) for t, idx in zip(part_features, cls_indexes)]
#         concatenated_features = torch.cat([torch.cat([c2d, s1d.expand_as(c2d)], dim=1) for c2d, s1d in zip(component_features, shape_features)])

#         #getting obb_features
#         obb_features = batch['obb_corners']
#         obb_features = [torch.cat([shape_obbs[:cls_idx],shape_obbs[cls_idx+1:]]).float().to(self.device) for shape_obbs, cls_idx in zip(obb_features, cls_indexes)]
#         obb_features = [self.encode_obb(shape_obbs) for shape_obbs in obb_features]
#         obb_features = torch.cat(obb_features)
        

#         #print(f'{concatenated_features.shape=}')
#         #getting motion vectors
#         # motion_vectors = [[ cv  for j,cv  in enumerate(sv) if cls_indexes[i] !=j  ]  for i,  sv in enumerate(batch['vectors'])] # removing the cls vectors
#         # vector_tensors = torch.from_numpy(np.vstack([cv for sv in motion_vectors for cv in sv],dtype=np.float64)).float()
#         # num_vectors = [[ cv.shape[0]  for j,cv  in enumerate(sv) if cls_indexes[i] !=j  ]  for i,  sv in enumerate(batch['vectors'])]
#         # num_vectors = torch.tensor([cv for sv in num_vectors for cv in sv]).to(self.device)
#         num_vectors = batch['num_vectors'].to(self.device)
#         input_vectors = batch['vectors'][:,:7].to(self.device)
#         labels = batch['vectors'][:,-1].to(self.device)

#         combined_features = self.shape_pro(concatenated_features)
#         combined_features = combined_features.repeat_interleave(num_vectors,dim=0)
#         obb_features = obb_features.repeat_interleave(num_vectors,dim=0)
#         vector_features = self.mo_param_pro(input_vectors)
#         final_features = torch.cat([combined_features,vector_features,obb_features],dim=1)
#         predictions = self.f_non_linear(self.final_projection(final_features))
#         predictions = predictions.reshape((-1,))

#         return predictions, labels
# the end



    # def batch_vectors(self,features,offsets):
    #     max_values = torch.zeros((self.max_sequence_length,self.input_self_attention_channel))
    #     for i in range(len(offsets) - 1):
    #         group = features[offsets[i]:offsets[i+1]]
    #         max_values[i]=torch.max(group, dim=0).values
    #     attention_mask = torch.ones((self.max_sequence_length, self.max_sequence_length), dtype=torch.int)
    #     attention_mask[:,len(offsets)-1:] = 0
    #     print(max_values.is_leaf)
    #     return max_values,attention_mask

    # # def batch_padded_tensors(self,batch_features):
    # #     masks = []
    # #     for i,feature_tensors in enumerate(batch_features):
            
    # #         padded_src = torch.nn.functional.pad(feature_tensors, (0,0,0,self.max_sequence_length - feature_tensors.shape[0]), value=0)
    # #         batch_features[i] = padded_src
    # #         attention_mask = torch.ones((self.max_sequence_length, self.max_sequence_length), dtype=torch.int)
    # #         attention_mask[:,feature_tensors.shape[0]:] = 0
    # #         masks.append(attention_mask)

    # #     return torch.tensor(batch_features),torch.tensor(masks)
















#   # batch_features = torch.zeros((len(batch_list),self.max_sequence_length,self.input_self_attention_channel))
#         # masks = torch.ones((len(batch_list),self.max_sequence_length,self.max_sequence_length))
#         # for b,batch in enumerate(batch_list):
            
            
#         #     batch_features[b],masks[b] = self.batch_vectors(point_dict_features.feat,
#         #                                              nn.functional.pad(point_dict_features.offset,(1,0)))
        
#         # cls_features =  batch_features[torch.arange(batch_features.shape[0]),cls_indexes,:]
#         # print(cls_features)
#         # print(batch_features[0,0,:])
#         # print(batch_features[1,0,:])
        
        


# #impt do not delete
# # part_features_padded = [ nn.functional.pad(pa_feature,(0,0,0,self.max_sequence_length-pa_feature.shape[0])) for pa_feature in part_features]
# #         masks = [nn.functional.pad(torch.ones((self.max_sequence_length,pa_feature.shape[0])),(0,self.max_sequence_length-pa_feature.shape[0])) for pa_feature in part_features]
# #         part_features_padded, masks = torch.stack(part_features_padded),torch.stack(masks)        