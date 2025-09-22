import torch
from carticulate import ESKF
from utils.transform import IS2MB
from utils.downsample import deduplic, get_sampled_data_indices
from collections import Counter
import os
from datasets.MagIMU.align import start

def run_pipeline(file_path, net, detector=None, corrector=None):
    RMI, RSB, t, RIS, aS, wS, mS = deduplic(file_path)
    seq_name = os.path.basename(file_path)[:-3]
    start_k = start[seq_name]
    poses, trans = offline_run(t, RMI, RSB, RIS, aS, wS, mS, start_k, net, detector, corrector)
    return poses, trans

def offline_run(time_stamp_100, RMI, RSB, RIS_100, aS_100, wS_100, mS_100, start_k, net, detector=None, corrector=None):
    net.rnn_initialize()

    net_indices = get_sampled_data_indices(time_stamp_100)[start_k:]
    repeat_times = Counter(net_indices) # 可能由于网络延迟，某个i被重复读了若干次

    eskf = [ESKF(an=5e-2, wn=5e-3, aw=1e-4, ww=1e-5, mn=5e-3) for _ in range(6)]
    RIS_eskf = RIS_100[0].clone()
    for imu_idx in range(6):
        eskf[imu_idx].initialize_9dof(RIS=RIS_100[0][imu_idx], gI=[0, 0, 9.79], nI=[1., 0, 0])
    
    poses, trans = [], []
    k = start_k
    flag = torch.zeros(6, dtype=torch.bool)
    pose = torch.zeros(24, 3, 3).cuda()
    
    for i in range(1, len(time_stamp_100)):
        flag = detector(pose, mS_100[i])
            
        for imu_idx in range(6):
            eskf[imu_idx].predict(am=aS_100[i][imu_idx], wm=wS_100[i][imu_idx], dt=time_stamp_100[i]-time_stamp_100[i-1])
            if flag[imu_idx]:
                eskf[imu_idx].correct(am=aS_100[i][imu_idx], wm=wS_100[i][imu_idx], mm=mS_100[i][imu_idx])
            else:
                eskf[imu_idx].correct(am=aS_100[i][imu_idx], wm=wS_100[i][imu_idx])
            RIS_eskf[imu_idx] = torch.from_numpy(eskf[imu_idx].get_orientation_R())

        if not i in net_indices: continue
        for _ in range(repeat_times[i]):
            RMB_eskf, aM_eskf, wM_eskf = IS2MB(RIS_eskf, aS_100[i], wS_100[i], RMI, RSB)

            if corrector is not None:
                aM_correct, wM_correct, RMB_correct = corrector.forward_frame(aM_eskf.cuda(), wM_eskf.cuda(), RMB_eskf.cuda())
                pose, tran = net.forward_frame(aM_correct.cuda(), wM_correct.cuda(), RMB_correct.cuda())
            else:
                pose, tran = net.forward_frame(aM_eskf.cuda(), wM_eskf.cuda(), RMB_eskf.cuda())
            poses.append(pose.cpu())
            trans.append(tran.cpu())
            k+=1

            if corrector is not None:
                corrector.set_flag(any(flag==0))

    poses = torch.stack(poses)
    trans = torch.stack(trans)
    return poses, trans