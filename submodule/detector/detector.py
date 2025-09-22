from abc import ABC, abstractmethod
import torch
import articulate as art



class BaseDetector(ABC):      
    @abstractmethod
    def __call__(self, pose, mM, RSB=None):
        pass

class BasicNormDetector(BaseDetector):
    def __init__(self, eps=0.15):
        self.eps = eps
        self.upper_bound = 1 + self.eps
        self.lower_bound = 1 - self.eps

    def __call__(self, pose, mS):
        m_norm = torch.norm(mS, dim=1)
        flag = (m_norm >= self.lower_bound) & (m_norm <= self.upper_bound)
        return flag
    

class OurDetector(BaseDetector):
    def __init__(self, k=3, eps=0.15):
        self.k = k
        self.eps = eps
        self.upper_bound = 1 + self.eps
        self.lower_bound = 1 - self.eps
        self.bm = art.ParametricModel("models/SMPL_male.pkl")

    def __call__(self, pose, mS):
        _, j = self.bm.forward_kinematics(pose.reshape(1, 24, 3, 3).cpu())
        j = j[0, [20, 21, 4, 5, 15, 0]]
        m_norm = torch.norm(mS, dim=1)
        distances = torch.norm(j.unsqueeze(1) - j.unsqueeze(0), dim=2)
        _, topk_indices = torch.topk(distances, k=self.k, dim=1, largest=False)
        relevant_magnitudes = m_norm[topk_indices]  # shape: (6, k)
        in_range = (relevant_magnitudes >= self.lower_bound) & (relevant_magnitudes <= self.upper_bound)
        flag = torch.all(in_range, dim=1)
        return flag
    
    