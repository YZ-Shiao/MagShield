import torch
import numpy as np


def deduplic(measurements_file_path):
    RMI, RSB, t_list, RIS_list, aS_list, wS_list, mS_list = torch.load(measurements_file_path).values()
    t_ret, RIS_ret, aS_ret, wS_ret, mS_ret = [], [], [], [], []
    t_last = -1
    for i in range(len(t_list)):
        if t_list[i] != t_last:
            t_ret.append(t_list[i])
            RIS_ret.append(RIS_list[i])
            aS_ret.append(aS_list[i])
            wS_ret.append(wS_list[i])
            mS_ret.append(mS_list[i])
        t_last = t_list[i]
    return RMI, RSB, t_ret, RIS_ret, aS_ret, wS_ret, mS_ret


def get_sampled_data_indices(time_stamp):
    start_time = time_stamp[0]
    end_time = time_stamp[-1]
    num_frames = int((end_time - start_time) * 60)
    sampled_time_stamps = np.linspace(start_time, end_time, num_frames+1)

    data_indices = []
    data_index = 0
    for t in sampled_time_stamps:
        while data_index < len(time_stamp)-1 and abs(time_stamp[data_index] - t) > abs(time_stamp[data_index + 1] - t):
            data_index += 1
        data_indices.append(data_index)
    return data_indices

     

def downsample_60fps(time_stamp, **data):
    # 近邻采样
    sampled_data_indices = get_sampled_data_indices(time_stamp)
    sampled_data = {key: [] for key in data.keys()}
    for data_index in sampled_data_indices:
        for key in data:
            sampled_data[key].append(data[key][data_index])
    return sampled_data