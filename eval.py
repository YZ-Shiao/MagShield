import torch
import os
import articulate as art
import tqdm


class ReducedPoseEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator("models/SMPL_male.pkl", joint_mask=torch.tensor([1, 2, 16, 17]))
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

    def __call__(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100])


def evaluate_pose(gt_dir, result_dir):
    print(f"======{result_dir}======")
    os.makedirs(os.path.join(result_dir, "pose_error"), exist_ok=True)
    evaluator = ReducedPoseEvaluator()
    pose_errors = []
    for file in os.listdir(result_dir):
        if not file.endswith(".pt"): continue
        if os.path.exists(os.path.join(result_dir, "pose_error", file)):
            pose_error = torch.load(os.path.join(result_dir, "pose_error", file))
        else:
            pose_gt, _ = torch.load(os.path.join(gt_dir, file)).values()
            pose_predict = torch.load(os.path.join(result_dir, file))[0]
            end = min(len(pose_gt), len(pose_predict))
            pose_gt = pose_gt[: end]
            pose_predict = pose_predict[:end]
            pose_error = evaluator(pose_predict, pose_gt)
            torch.save(pose_error, os.path.join(result_dir, "pose_error", file))
        pose_errors.append(pose_error)

    pose_errors = torch.stack(pose_errors).mean(dim=0)
    for name, error in zip(evaluator.names, pose_errors):
        print('%s: %.4f' % (name, error[0]))



def evaluate_tran(gt_dir, result_dir, align=True):
    folder_name = "tran_error_align" if align else "tran_error"
    os.makedirs(os.path.join(result_dir, folder_name), exist_ok=True)

    for file in tqdm.tqdm(os.listdir(result_dir), ncols=50):
        tran_errors = {window_size: [] for window_size in list(range(1, 7))}

        if not file.endswith(".pt"): continue
        if os.path.exists(os.path.join(result_dir, folder_name, file)): continue

        pose_gt, tran_gt = torch.load(os.path.join(gt_dir, file)).values()
        pose_predict, tran_predict = torch.load(os.path.join(result_dir, file))

        end = min(len(tran_gt), len(tran_predict))
        tran_gt = tran_gt[: end].float()
        tran_predict = tran_predict[:end].float()

        move_distance_t = torch.zeros(tran_gt.shape[0])
        v = (tran_gt[1:] - tran_gt[:-1]).norm(dim=1)
        for j in range(len(v)):
            move_distance_t[j + 1] = move_distance_t[j] + v[j]

        for window_size in tran_errors.keys():
            frame_pairs = []
            start, end = 0, 1
            while end < len(move_distance_t):
                if move_distance_t[end] - move_distance_t[start] < window_size:
                    end += 1
                else:
                    if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                        frame_pairs.append((start, end))
                    start += 1

            # calculate mean distance error
            errs = []
            for start, end in frame_pairs:
                vel_p = tran_predict[end] - tran_predict[start]
                vel_t = tran_gt[end] - tran_gt[start]

                if align:
                    heading_gt = pose_gt[start][0, :, 2] * torch.tensor([1., 0., 1.])
                    heading_predict = pose_predict[start][0, :, 2] * torch.tensor([1., 0., 1.])

                    # align the heading of the predicted trajectory with the ground truth
                    dot = heading_gt.dot(heading_predict) / (heading_gt.norm() * heading_predict.norm())
                    if dot > 0.98:
                        angle = 0
                    elif dot < -0.98:
                        angle = torch.pi
                    else:
                        angle = torch.acos(dot)
                        if heading_gt.cross(heading_predict).dot(torch.tensor([0., 1., 0.])) < 0:
                            angle = -angle

                    rot = art.math.axis_angle_to_rotation_matrix(torch.tensor([0., angle, 0.])).reshape(3, 3)
                    vel_t = rot @ vel_t

                errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                assert not torch.isnan(errs[-1])

            if len(errs) > 0:
                tran_errors[window_size].append(sum(errs) / len(errs))

        torch.save(tran_errors, os.path.join(result_dir, folder_name, file))


def compare_tran_err():
    import matplotlib.pyplot as plt
    import torch

    result_dirs = [
        "results/PNP/eskf9/tran_error_align",
        "results/PNP/eskf9+det/tran_error_align",
        "results/PNP/eskf9+det+cor/tran_error_align",
    ]

    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.figure(dpi=200)
    plt.grid(linestyle='-.')
    plt.xlabel('Real travelled distance (m)', fontsize=16)
    plt.ylabel('Mean translation error (m)', fontsize=16)
    plt.title('Cumulative Translation Error (on MagIMU)', fontsize=18)
    
    for result_dir in result_dirs:
        tran_errors = {window_size: [] for window_size in list(range(1, 7))}
        for file in os.listdir(result_dir):
            tran_error = torch.load(os.path.join(result_dir, file))
            for window_size, error in tran_error.items():
                if window_size in tran_errors.keys(): 
                    tran_errors[window_size].append(error)
        
        plt.plot([0] + [_ for _ in tran_errors.keys()], [0] + [torch.tensor(_).mean() for _ in tran_errors.values()])
    plt.legend(['eskf', 'eskf+det', 'eskf+det+cor'], fontsize=15)
    plt.show()
    plt.cla()


if __name__ == '__main__':
    # evaluate_pose("datasets/MagIMU/gt", "results/PNP/eskf9")
    # evaluate_pose("datasets/MagIMU/gt", "results/PNP/eskf9+det")
    # evaluate_pose("datasets/MagIMU/gt", "results/PNP/eskf9+det+cor")
    # evaluate_tran("datasets/MagIMU/gt", "results/PNP/eskf9")
    # evaluate_tran("datasets/MagIMU/gt", "results/PNP/eskf9+det")
    # evaluate_tran("datasets/MagIMU/gt", "results/PNP/eskf9+det+cor")
    compare_tran_err()
    # evaluate_pose("datasets/MagIMU/gt", "results/DynaIP/eskf9")
    # evaluate_pose("datasets/MagIMU/gt", "results/DynaIP/eskf9+det")
    # evaluate_pose("datasets/MagIMU/gt", "results/DynaIP/eskf9+det+cor")




 