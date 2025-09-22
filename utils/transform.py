import torch

def IS2MB(RIS, aS, wS, RMI, RSB):
    aI = RIS.matmul(aS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., 9.8])
    aM = RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
    wI = RIS.matmul(wS.unsqueeze(-1)).squeeze(-1)
    wM = RMI.matmul(wI.unsqueeze(-1)).squeeze(-1)
    RMB = RMI.matmul(RIS).matmul(RSB)
    return RMB, aM, wM