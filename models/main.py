import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn.utils.rnn import pad_sequence
from .Point_transformer_V3 import Point,PointTransformerV3
from torch.nn import MultiheadAttention
from .transformer_encoder_layer import TransformerEncoderLayerCustom
import torch_scatter
from .DGCNN import DGCNN_cls
import numpy as np




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
    def __init__(self, input_size, channels, act_layer=nn.Tanh,last_layer_act=True,drop_prob=0.1):
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

        for i,layer in enumerate(self.layers):
            prev_data = input_data
            input_data = layer(input_data)
        return input_data,prev_data

# takes 2 tensor
class Encode_Obb(nn.Module):
    def __init__(self, input_size, channels, ):
        super(Encode_Obb,self).__init__()
        self.pp =  MLP(input_size=3,channels=channels)

    def forward(self, corners):
        corners_features = self.pp(corners)
        return torch.max(corners_features,dim=-2).values



class Predictor(nn.Module):
    def __init__(
        self,
        device,
        features_reduce='max',
        grid_size=0.01,
        backbone_out_channels=(32, 64, 128, 256, 512),
        max_seq_len = 50,
        num_heads = 32,
        num_layers = 1,
        cls_token = 'cls',
        enable_flash=True,

    ):
        super(Predictor,self).__init__()
        self.device = device
        self.features_reduce = features_reduce
        self.grid_size = grid_size
        self.max_seq_len = max_seq_len
        self.cls_token = cls_token
        self.input_self_attention_channel = backbone_out_channels[-1]

        #self.backbone = PointTransformerV3(enc_channels=backbone_out_channels,enable_flash = enable_flash)
        self.backbone = DGCNN_cls(512)
        self.encode_obb  = Encode_Obb(input_size=3,channels=[256,512,512])
        #self.transformer =  Transformer(source_max_seq_len=max_sequence_length,embedding_dim=backbone_out_channels[-1],num_heads=num_heads,num_layers=num_layers)
        self.main_mlp = MLP(input_size=2*backbone_out_channels[-1],
                            channels=[int(3*backbone_out_channels[-1]),1024])
        
        self.motion_type_mlp = MLP(input_size=1024,
                            channels=[2048,256,128,4],
                            last_layer_act=False)
        
        self.orientation_type_mlp = MLP(input_size=1024,
                            channels=[2048,128,3],
                            last_layer_act=False)
        
        self.residual_mlp = MLP(input_size=1024,
                            channels=[2048,128,3*3],
                            last_layer_act=False)
        
        self.rotation_point_mlp = MLP(input_size=1024,
                            channels=[2048,128,3],
                            last_layer_act=False)
        


        


    def forward(self, batch_dict):
        
        batch_size = batch_dict['part_coord'].shape[0]
        pc_size = batch_dict['part_coord'].shape[1]
        

        # using ptv3 impt do not delete
        # part_offsets = torch.tensor([pc_size*(i+1) for i in range(batch_size)],dtype=torch.int)
        # shape_offsets = torch.tensor([pc_size*(i+1) for i in range(batch_size)],dtype=torch.int)
        # part_coord, shape_coord = batch_dict['part_coord'].reshape(-1,3), batch_dict['shape_coord'].reshape(-1,3)
        # part_input = dict(  coord= part_coord.to(self.device),
        #                     feat= part_coord.to(self.device),
        #                     offset= part_offsets.to(self.device),
        #                     grid_size=self.grid_size)
        # shape_input = dict(  coord= shape_coord.to(self.device),
        #                     feat= shape_coord.to(self.device),
        #                     offset= shape_offsets.to(self.device),
        #                     grid_size=self.grid_size)
        # part_out = self.backbone(part_input)
        # shape_out = self.backbone(shape_input)
        # features = part_out.feat
        # offsets = part_out.offset
        # offsets = nn.functional.pad(offsets,(1,0))
        # part_features = torch_scatter.segment_csr(src=features,indptr=offsets,reduce=self.features_reduce)
        # features = shape_out.feat
        # offsets = shape_out.offset
        # offsets = nn.functional.pad(offsets,(1,0))
        # shape_features = torch_scatter.segment_csr(src=features,indptr=offsets,reduce=self.features_reduce)

        #using dgcnn
        part_coord = batch_dict['part_coord']
        part_coord = part_coord.view(part_coord.shape[0],3,-1).to(self.device)
        shape_coord = batch_dict['shape_coord']
        shape_coord = shape_coord.view(shape_coord.shape[0],3,-1).to(self.device)
        part_features,shape_features = self.backbone(part_coord),self.backbone(shape_coord)

        


        # obb_corners = batch_dict['obb_corners']
        # obb_features = self.encode_obb(obb_corners.to(self.device))

        comibined_features = torch.cat([part_features,shape_features],dim=1)
        comibined_features = self.main_mlp(comibined_features)
        
        predicted_motion_type = self.motion_type_mlp(comibined_features)
        predicted_orientation_label = self.orientation_type_mlp(comibined_features)
        predicted_residual = self.residual_mlp(comibined_features).reshape(-1,3,3)
        predicted_rotation_point = self.rotation_point_mlp(comibined_features)

        output = dict(
            predicted_motion_type=predicted_motion_type,
            predicted_orientation_label=predicted_orientation_label,
            predicted_residual= predicted_residual,
            predicted_rotation_point= predicted_rotation_point,
            g_motion_type = batch_dict['motion_type'].to(self.device),
            g_orientation_label = batch_dict['orientation_label'].to(self.device),
            g_residual = batch_dict['residual'].to(self.device),
            g_rotation_point = batch_dict['rotation_point'].to(self.device),
            g_truth_axis = batch_dict['g_truth_axis'].to(self.device),
            paths=batch_dict['paths']
        )
        return output


        




class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, batch_first: bool, num_layers=2):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead,dim_feedforward=dim_feedforward,batch_first=batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)


    def forward(self, src: Tensor, src_key_padding_mask:Tensor) -> Tensor:
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
    

        
        
        


class Predictor_Transformer(nn.Module):
    def __init__(
        self,
        device,
        features_reduce='max',
        grid_size=0.01,
        backbone_out_channels=(32, 64, 128, 256, 512),
        max_seq_len = 50,
        embedding_dim=512,
        nhead=8,
        encode_part=True,
        encode_shape=True,
        num_layers = 2,
        dim_feedforward=2048,
        enable_flash=True,

    ):
        super(Predictor_Transformer,self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        #self.features_reduce = features_reduce
        self.grid_size = grid_size
        self.max_seq_len = max_seq_len
        self.nhead=nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.encode_part = encode_part
        self.encode_shape = encode_shape
        self.batch_first = True
        self.input_self_attention_channel = backbone_out_channels[-1]

        # self.backbone = PointTransformerV3(enc_channels=backbone_out_channels,enable_flash = enable_flash)
        self.backbone = DGCNN_cls(embedding_dim,encode_part=encode_part)
        #self.encode_obb  = Encode_Obb(input_size=3,channels=[256,512,512])
        if encode_part:
            self.part_encoder = TransformerEncoderLayerCustom(d_model=self.embedding_dim,nhead=nhead,dim_feedforward=dim_feedforward,batch_first=self.batch_first)
            self.cls_token = torch.nn.Parameter(
                    torch.randn(1, 1, self.embedding_dim)
                )  
        if encode_shape:
            self.shape_encoder =  TransformerModel(d_model=embedding_dim,nhead=nhead,
                                                dim_feedforward=dim_feedforward,batch_first=self.batch_first,num_layers=num_layers)
  
        
        self.main_mlp = MLP(input_size=2*backbone_out_channels[-1],
                            channels=[int(3*backbone_out_channels[-1]),1024])
        
        self.motion_type_mlp = MLP(input_size=self.embedding_dim,
                            channels=[2048,256,128,4],
                            last_layer_act=False)
        
        self.orientation_type_mlp = MLP(input_size=self.embedding_dim,
                            channels=[2048,128,3],
                            last_layer_act=False)
        
        self.residual_mlp = MLP(input_size=self.embedding_dim,
                            channels=[2048,128,3*3],
                            last_layer_act=False)
        
        self.rotation_point_mlp = MLP(input_size=self.embedding_dim,
                            channels=[2048,128,3*3],
                            last_layer_act=False)
        


        


    def forward(self, batch_dict):
        batch_size = batch_dict['num_parts'].shape[0]
        pc_size = batch_dict['part_coord'].shape[1]
        part_coord = batch_dict['part_coord']
        part_coord = part_coord.view(part_coord.shape[0],3,-1).to(self.device)# change the shape here, but it returns normally
        point_features = self.backbone(part_coord)

        # it splits according to the indices and the rest are one whatever are left
        if self.encode_part:
            part_features = self.part_encoder(query=self.cls_token.expand(point_features.shape[0],-1,-1),
                                              key=point_features,
                                              value=point_features)
            part_features = part_features.squeeze(1)

        else:
            part_features = point_features

        part_features = torch.tensor_split(part_features,batch_dict['split_indices'][:-1])
        padded_part_features = pad_sequence(part_features,batch_first=self.batch_first,padding_value=0)
        encoder_mask = ~ (padded_part_features!=0).any(dim=-1)
        #TODO have to debug this
        if self.encode_shape:
            transformed_features = self.shape_encoder(src=padded_part_features,src_key_padding_mask=encoder_mask)
        else:
            transformed_features = padded_part_features
        
        features_mask = torch.arange(transformed_features.shape[1]).unsqueeze(0).expand(batch_size,-1).clone().to(self.device) < batch_dict['num_parts'].unsqueeze(1).to(self.device)
        features_mask[:,0] = False # batch_size,sequence_dim
  
        transformed_features = transformed_features.reshape(-1,self.embedding_dim)#total_padded_parts(including_cls),embeddin_dim 
        features_mask = features_mask.reshape(-1)#total_padded_parts(including_cls)
        transformed_features = transformed_features[features_mask]#total_non_padded_parts,excluding_cls
        ground_truth_mask = torch.ones(batch_dict['split_indices'][-1]).bool()
        ground_truth_mask[0] = False
        ground_truth_mask[batch_dict['split_indices'][:-1]]=False# for ground truth masks, this also does not include the cls
     
        g_motion_type = batch_dict['motion_type'][ground_truth_mask].to(self.device)
        g_orientation_label = batch_dict['orientation_label'][ground_truth_mask].to(self.device)
        g_residual = batch_dict['residual'][ground_truth_mask].to(self.device)
        g_rotation_point = batch_dict['rotation_point'][ground_truth_mask].to(self.device)
        g_truth_axis = batch_dict['g_truth_axis'][ground_truth_mask].to(self.device)
        movable_ids = batch_dict['movable_id'][ground_truth_mask].to(self.device)
        movable = batch_dict['movable'][ground_truth_mask].to(self.device)
        
        
        # comibined_features = torch.cat([part_features,transformed_features],dim=1)
        # comibined_features = self.main_mlp(comibined_features)
        
        predicted_motion_type,_ = self.motion_type_mlp(transformed_features)
        predicted_orientation_label,_= self.orientation_type_mlp(transformed_features)
        predicted_residual, _ = self.residual_mlp(transformed_features)
        predicted_residual = predicted_residual.reshape(-1,3,3)
        predicted_rotation_point,_ = self.rotation_point_mlp(transformed_features)
        predicted_rotation_point = predicted_rotation_point.reshape(-1,3,3)

        output = dict(
            movable=movable,
            movable_ids =movable_ids,
            predicted_motion_type=predicted_motion_type,
            predicted_orientation_label=predicted_orientation_label,
            predicted_residual= predicted_residual,
            predicted_rotation_point= predicted_rotation_point,
            g_motion_type = g_motion_type,
            g_orientation_label = g_orientation_label,
            g_residual = g_residual,
            g_rotation_point = g_rotation_point,
            g_truth_axis =  g_truth_axis,
            paths=batch_dict['paths'],
            num_parts = batch_dict['num_parts']
        )
        return output
        
        
       































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