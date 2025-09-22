import articulate as art
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import *
from torch.nn.functional import relu


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=False, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=False)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h=None):
        length = [_.shape[0] for _ in x]
        x = self.dropout(F.relu(self.linear1(pad_sequence(x))))
        x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), h)[0]
        x = self.linear2(pad_packed_sequence(x)[0])
        return [x[:l, i].clone() for i, l in enumerate(length)]
   
class RNNWithInit(RNN):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_init: int, n_rnn_layer: int
                 , bidirectional=False, dropout=0.2):
        super().__init__(n_input, n_output, n_hidden, n_rnn_layer, bidirectional, dropout)
        self.n_rnn_layer = n_rnn_layer
        self.n_hidden = n_hidden
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(n_init, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
        )

    def forward(self, x, _=None):
        x, x_init = x
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(x_init).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, (h, c))

class SubPoser(nn.Module):
    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0):
        super(SubPoser, self).__init__()
        
        self.extra_dim = extra_dim
        self.rnn1 = RNNWithInit(n_init=v_output, n_input=n_input-extra_dim, 
                                n_hidden=n_hidden, n_output=v_output, 
                                n_rnn_layer=num_layer, dropout=dropout)
        self.rnn2 = RNNWithInit(n_init=p_output, n_input=n_input+v_output,
                                n_hidden=n_hidden, n_output=p_output, 
                                n_rnn_layer=num_layer, dropout=dropout)

        self.rnn_states =[None, None]


    def forward(self, x, v_init, p_init):
        if self.extra_dim != 0:
            x_v = [_[:, :-self.extra_dim] for _ in x] # remove glb information when predict local part velocity
            v = self.rnn1((x_v, v_init))
        else:
            v = self.rnn1((x, v_init))
        p = self.rnn2(([torch.cat(_, dim=-1) for _ in zip(x, v)], p_init))
        return v, p
    

    @torch.no_grad()
    def initialize(self, v_init, p_init):
        self.rnn_states[0] = [_.contiguous() for _ in self.rnn1.init_net(v_init).view(1, 2, self.rnn1.rnn.num_layers, self.rnn1.rnn.hidden_size).permute(1, 2, 0, 3)]
        self.rnn_states[1] = [_.contiguous() for _ in self.rnn2.init_net(p_init).view(1, 2, self.rnn2.rnn.num_layers, self.rnn2.rnn.hidden_size).permute(1, 2, 0, 3)]


    @torch.no_grad()
    def forward_frame(self, imu):
        imu = imu.unsqueeze(0)

        if self.extra_dim != 0:
            x = imu[:, :-self.extra_dim]
        else:
            x = imu

        x, self.rnn_states[0] = self.rnn1.rnn(relu(self.rnn1.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[0])
        x = self.rnn1.linear2(x[0])
        v = x.clone().flatten()

        x = torch.cat([imu, x], dim=1)
        x, self.rnn_states[1] = self.rnn2.rnn(relu(self.rnn2.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[1])
        x = self.rnn2.linear2(x[0])
        p = x.clone().flatten()

        return v, p


        
    
    
class DynaIP(nn.Module):
    name = 'DynaIP'
    def __init__(self):
        super(DynaIP, self).__init__()

        n_hidden = 200
        num_layer = 2
        dropout = 0.2
        n_glb = 6
        
        self.posers = nn.ModuleList([SubPoser(n_input=36 + n_glb, v_output=6, p_output=24,
                                            n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), 
                                    SubPoser(n_input=48 + n_glb, v_output=12, p_output=12,
                                            n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb), 
                                    SubPoser(n_input=24 + n_glb, v_output=6, p_output=30,
                                            n_hidden=n_hidden, num_layer=num_layer, dropout=dropout, extra_dim=n_glb)])        
        
        self.glb = RNN(n_input=72, n_output=n_glb, n_hidden=36, n_rnn_layer=1, dropout=dropout) 
                
        self.sensor_names = ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']
        self.v_names = ['Root', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']
        self.p_names = ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'L3', 
                        'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'LeftUpperArm', 
                        'RightUpperArm']
        
        self.generate_indices_list()

        self.rnn_states = [None]
        self.bm = art.ParametricModel("models/SMPL_male.pkl")

        self.load_state_dict(torch.load('inertial_poser/DynaIP/weights/DynaIP_s.pth')) # DynaIP* in paper
       
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def find_indices(self, elements, lst):
        indices = []
        for element in elements:
            if element in lst:
                indices.append(lst.index(element))
        return indices
    
    def generate_indices_list(self):
        posers_config = [
            {'sensor': ['Root', 'LeftForeArm', 'RightForeArm'], 'velocity': ['LeftHand', 'RightHand'], 
             'pose': ['LeftShoulder', 'LeftUpperArm', 'RightShoulder', 'RightUpperArm']},   
            
            {'sensor': ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head'], 'velocity': ['Root', 'LeftFoot', 'RightFoot', 'Head'], 
             'pose': ['LeftUpperLeg', 'RightUpperLeg']},
                        
            {'sensor': ['Root', 'Head'], 'velocity': ['Root', 'Head'], 
             'pose': ['L5', 'L3', 'T12', 'T8', 'Neck']},
            
        ]        
        self.indices = []
        for i in range(len(self.posers)):
            temp = {'sensor_indices': self.find_indices(posers_config[i]['sensor'], self.sensor_names), 
                    'v_indices': self.find_indices(posers_config[i]['velocity'], self.v_names), 
                    'p_indices': self.find_indices(posers_config[i]['pose'], self.p_names)}
            self.indices.append(temp)
        
    def forward(self, x, v_init, p_init):
        r"""
        Args:
            x : List of tensors in shape (time, 6, 12)
            ['Root', 'LeftLowerLeg', 'RightLowerLeg', 'Head', 'LeftForeArm', 'RightForeArm']
            
            v_init (torch.Tensor): initial velocity of specific joints in (batch, 6, 3)
            ['Root', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot']
            
            p_init (torch.Tensor): initial pose tensor in shape (batch, 11, 6)
            ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'L3', 'T12', 'T8', 'Neck',
            'LeftShoulder', 'RightShoulder', 'LeftUpperArm', 'RightUpperArm']                 
        Returns:
            p_out: [7, 9, 8, 10, 0, 1, 2, 3, 4, 5, 6] index corresponding to p_init and posers_config
            v_out: [2, 3, 0, 4, 5, 1, 0, 1] index corresponding to v_init and posers_config  
        """
        v_out, p_out = [], []
        s_glb = self.glb([_.flatten(1) for _ in x])
        for i in range(len(self.posers)):
            sensor = [_[:, self.indices[i]['sensor_indices']].flatten(1) for _ in x]
            si = [torch.cat((l, g), dim=-1) for l, g in zip(sensor, s_glb)]
            vi = v_init[:, self.indices[i]['v_indices']].flatten(1)
            pi = p_init[:, self.indices[i]['p_indices']].flatten(1)
            v, p = self.posers[i](si, vi, pi)
            v_out.append(v)
            p_out.append(p)
        
        v_out = [torch.cat(_, dim=-1) for _ in zip(*v_out)]
        p_out = [torch.cat(_, dim=-1) for _ in zip(*p_out)]    
        
        return v_out, p_out    
    

    @torch.no_grad()
    def rnn_initialize(self, v_init=None, p_init=None):
        if v_init is None:
            v_init = torch.zeros(6, 3).cuda()
        if p_init is None:
            p_init = torch.eye(3)[:, :2].t().reshape(6).repeat(11, 1).cuda()

        for i in range(len(self.posers)):
            vi = v_init[self.indices[i]['v_indices']].flatten(0)
            pi = p_init[self.indices[i]['p_indices']].flatten(0)
            self.posers[i].initialize(vi, pi)
        self.rnn_states = [None]


    @torch.no_grad()
    def forward_frame(self, aM, wM, RMB):
        imu = PIP2DynaIP(aM, RMB).reshape(6, 12)  # (6, 12)
        x, self.rnn_states[0] = self.glb.rnn(relu(self.glb.linear1(imu.reshape(72)), inplace=True).unsqueeze(0), self.rnn_states[0])
        s_glb = self.glb.linear2(x[0])

        v_out, p_out = [], []
        for i in range(len(self.posers)):
            sensor = imu[self.indices[i]['sensor_indices']].flatten()     # (36)
            si = torch.cat((sensor, s_glb), dim=-1)
            v, p = self.posers[i].forward_frame(si)
            v_out.append(v)
            p_out.append(p)

        v_out = torch.cat(v_out, dim=-1)
        p_out = torch.cat(p_out, dim=-1)

        pose = p_out.cpu()
        pose = pose.view(-1, 11, 6)[:, [4, 5, 6, 7, 8, 9, 10, 0, 2, 1, 3]] 
        orientation = imu[:, :9].view(-1, 6, 3, 3).cpu()
        glb_full_pose_xsens = self._reduced_glb_6d_to_full_glb_mat_xsens(pose, orientation)
        glb_full_pose_smpl = self._glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens)   # (24, 3, 3)
        local_pose_smpl = self.bm.inverse_kinematics_R(glb_full_pose_smpl)  # (24, 3, 3)
        return local_pose_smpl.reshape(24, 3, 3), torch.zeros(3)


    
    def _reduced_glb_6d_to_full_glb_mat_xsens(self, glb_reduced_pose, orientation):
        joint_set = [19, 15, 1, 2, 3, 4, 5, 11, 7, 12, 8]
        sensor_set = [0, 20, 16, 6, 13, 9]
        ignored = [10, 14, 17, 18, 21, 22]
        parent = [9, 13, 16, 16, 20, 20]
        root_rotation = orientation[:, 0].view(-1, 3, 3)
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, len(joint_set), 3, 3)
        # back to glb coordinate
        glb_reduced_pose = root_rotation.unsqueeze(1).matmul(glb_reduced_pose)
        orientation[:, 1:] = root_rotation.unsqueeze(1).matmul(orientation[:, 1:])
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 23, 1, 1)
        global_full_pose[:, joint_set] = glb_reduced_pose
        global_full_pose[:, sensor_set] = orientation
        global_full_pose[:, ignored] = global_full_pose[:, parent]
        return global_full_pose    
    

    def _glb_mat_xsens_to_glb_mat_smpl(self, glb_full_pose_xsens):
        glb_full_pose_smpl = torch.eye(3).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
        indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
        for idx, i in enumerate(indices):
            glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]            
        return glb_full_pose_smpl


def PIP2DynaIP(aM, RMB):
    # RMB_PIP: (6, 3, 3), aM_PIP: (6, 3), lw, rw, lk, rk, h, r
    # RMB_DynaIP: (6, 3, 3), aM_DynaIP: (6, 3), r, lk, rk, h, lw, rw
    RMB_DynaIP = RMB[[5, 2, 3, 4, 0, 1]]
    aM_DynaIP = aM[[5, 2, 3, 4, 0, 1]]
    return normalize_imu(aM_DynaIP, RMB_DynaIP)

def normalize_imu(acc, ori):
    r"""
    normalize imu w.r.t the root sensor
    """
    acc = acc.view(-1, 6, 3)
    ori = ori.view(-1, 6, 3, 3)
    acc = torch.cat((acc[:, :1], acc[:, 1:] - acc[:, :1]), dim=1).bmm(ori[:, 0])
    ori = torch.cat((ori[:, :1], ori[:, :1].transpose(2, 3).matmul(ori[:, 1:])), dim=1)
    data = torch.cat((ori.view(-1, 6, 9), acc), dim=-1)
    return data
