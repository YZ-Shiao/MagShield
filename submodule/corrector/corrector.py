from articulate.utils.torch.rnn import *
import torch
from torch.nn.functional import relu
import articulate as art

def normalize_and_concat_reduced(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = (glb_acc[:, :5] - glb_acc[:, 5:]).bmm(glb_rot[:, -1])
    ori = glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5])
    gM = torch.zeros(len(glb_acc), 3, 1).to(glb_acc.device)
    gM[:, 1, 0] = -10
    gR = glb_rot[:, 5].transpose(1, 2).matmul(gM)
    data = torch.cat((gR.flatten(1), acc.flatten(1), ori.flatten(1)), dim=1)
    return data

def rotation_along_yaw_axis(aM, wM, RMB, angle):
    assert aM.shape == wM.shape == (6, 3)
    assert RMB.shape == (6, 3, 3)
    assert angle.shape == (6, )

    ZM = torch.tensor([0.0, 1.0, 0.0]).to(aM.device)
    transfer_aa = ZM * angle.reshape(6, 1)     # (6, 3)
    transfer_rot = art.math.axis_angle_to_rotation_matrix(transfer_aa).reshape(6, 3, 3)   # (6, 3, 3)

    aM_rotated = transfer_rot.matmul(aM.unsqueeze(-1)).squeeze(-1)   # (6, 3)
    wM_rotated = transfer_rot.matmul(wM.unsqueeze(-1)).squeeze(-1)   # (6, 3)
    RMB_rotated = transfer_rot.matmul(RMB)   # (6, 3, 3)
    return aM_rotated, wM_rotated, RMB_rotated


class CorrectorV2(torch.nn.Module):
    def __init__(self):
        super(CorrectorV2, self).__init__()
        self.rnn = RNN(input_size=63,
                       output_size=10,
                       hidden_size=256,
                       num_rnn_layer=2,
                       dropout=0.4)


        self.load_state_dict(torch.load('submodule/corrector/weights.pt'))
        self.rnn_state = None
        self.weight = torch.zeros(1)
        self.use_flag = False

    def set_flag(self, flag: bool):
        self.use_flag = flag
        
    def forward(self, x):
        # for training only
        return self.rnn(x)
    
    @torch.no_grad()
    def forward_frame(self, aM, wM, RMB):
        self.weight = self.weight+1/90 if self.use_flag else self.weight-1/90
        self.weight = torch.clamp(self.weight, 0, 1)

        input_data = normalize_and_concat_reduced(aM, RMB)
        x, self.rnn_state = self.rnn.rnn(relu(self.rnn.linear1(input_data), inplace=True).unsqueeze(0), self.rnn_state)
        x = self.rnn.linear2(x[0])

        x = x.reshape(10)
        embed_cos = x[:5]
        embed_sin = x[5:]
        embed_points = torch.stack((embed_cos, embed_sin), dim=-1)
        embed_points = embed_points / embed_points.norm(dim=-1, keepdim=True)
        angle_out = torch.atan2(embed_points[:, 1], embed_points[:, 0]).reshape(5)
        
        angle_predict_leaf = angle_out * self.weight.to(angle_out.device)
        angle_predict = torch.cat((angle_predict_leaf, torch.zeros(1).to(angle_out.device)), dim=-1)
        
        aM_correct, wM_correct, RMB_correct = rotation_along_yaw_axis(aM, wM, RMB, -angle_predict)
        return aM_correct, wM_correct, RMB_correct