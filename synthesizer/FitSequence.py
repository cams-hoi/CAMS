import numpy as np
import torch
import trimesh
import trimesh.proximity
from scipy.spatial.transform import Rotation as R
import pickle
from tqdm import tqdm

from core.utils.amano import ManoLayer as AManoLayer

mano_layer = AManoLayer(cuda = True)

def get_stage_fitting_target(
    ref_flag_1:torch.Tensor,
    ref_flag_2:torch.Tensor,
    ref_1:torch.Tensor,
    ref_2:torch.Tensor,
    seq_1:torch.Tensor,
    seq_2:torch.Tensor,
    n_flag:torch.Tensor,
    stage_length,
    objs,
    obj_traj:torch.Tensor
):
    # Compute coefs regarding to the two ends of the stage
    loss_coef_1 = torch.zeros_like(n_flag)
    loss_coef_2 = torch.zeros_like(n_flag)
    len_frame = n_flag.shape[1]
    part_cnt = n_flag.shape[0]
    for frame_i in range(stage_length):
        for part_i in range(n_flag.shape[0]):
            for finger_i in range(n_flag.shape[2]):
                if ref_flag_1[part_i, finger_i] < 0.5 and ref_flag_2[part_i, finger_i] < 0.5:
                    continue
                elif ref_flag_2[part_i, finger_i] < 0.5:
                    loss_coef_1[part_i, frame_i, finger_i] = 1
                elif ref_flag_1[part_i, finger_i] < 0.5:
                    loss_coef_2[part_i, frame_i, finger_i] = 1
                else:
                    loss_coef_1[part_i, frame_i, finger_i] = 1-frame_i/stage_length
                    loss_coef_2[part_i, frame_i, finger_i] = frame_i/stage_length

    n_flag = torch.where(n_flag < 0.5, 0, 1)
    loss_coef_1 *= n_flag
    loss_coef_2 *= n_flag

    # Compute optimize target
    def get_target(seq, ref):
        tip_target = torch.zeros((part_cnt, len_frame, 5, 3), dtype=torch.float32).cuda()
        joint_target = torch.zeros((part_cnt, len_frame, 5, 4, 3), dtype=torch.float32).cuda()
        for part_i in range(part_cnt):
            for finger_i in range(5):
                tip_target_pf = seq[part_i, :, finger_i, 12:15] + ref[part_i, finger_i, 0:3]
                joint_target_pf = seq[part_i, :, finger_i, 0:12].reshape(-1, 4, 3)
                tip_target[part_i, :, finger_i] = torch.from_numpy(R.from_rotvec(obj_traj[part_i, :, 3:6].cpu()).apply(tip_target_pf.cpu())).cuda() + obj_traj[part_i, :, 0:3]
                for s in range(4):
                    joint_target[part_i, :, finger_i, s] = torch.from_numpy(R.from_rotvec(obj_traj[part_i, :, 3:6].cpu()).apply(joint_target_pf[:, s].cpu())).cuda()
        return tip_target, joint_target

    tip_target_1, joint_target_1 = get_target(seq_1, ref_1)
    tip_target_2, joint_target_2 = get_target(seq_2, ref_2)

    return tip_target_1, joint_target_1, loss_coef_1, tip_target_2, joint_target_2, loss_coef_2

def fit_joint(
    tip_target:torch.Tensor,
    joint_target:torch.Tensor,
    loss_coef:torch.Tensor,
    init_trans,
    init_pose
):
    len_frame = tip_target.shape[0]
    part_cnt = tip_target.shape[1]
    trans_t = torch.zeros((len_frame-1, 3), device='cuda')
    pose_t = torch.zeros((len_frame-1, 48), device='cuda')
    trans_t[:] = init_trans
    pose_t[:] = init_pose
    trans_t.requires_grad_(True)
    pose_t.requires_grad_(True)
    opt = torch.optim.Adam([trans_t, pose_t], lr=0.05)

    # Optimization
    for i in tqdm(range(2000)):
        opt.zero_grad()
        trans = torch.cat([init_trans.unsqueeze(0), trans_t], dim=0)
        pose = torch.cat([init_pose.unsqueeze(0), pose_t], dim=0)
        mano_output = mano_layer(pose[:, :3], pose[:, 3:])
        joints = mano_output.joints - mano_output.joints[:, :1] + trans.unsqueeze(1)
        loss_t, loss_j = 0, 0
        for part_i in range(part_cnt):
            for finger_i in range(5):
                finger_p = torch.cat([joints[:, :1], joints[:, finger_i*4+1:finger_i*4+5]], dim=1)
                tip_p = finger_p[:, 4]
                joint_p = finger_p[:, :4] - finger_p[:, 4:]
                joint_p = joint_p / torch.norm(joint_p, dim=2, keepdim=True)
                loss_t += torch.sum(torch.sum(torch.square(tip_p-tip_target[:, part_i, finger_i]), dim=1) * loss_coef[:, part_i, finger_i])
                loss_j += torch.sum(torch.sum(torch.square(joint_p-joint_target[:, part_i, finger_i]), dim=(1,2)) * loss_coef[:, part_i, finger_i])
                # finger_p = torch.cat([joints[:, :1], joints[:, finger_i*4+1:finger_i*4+5]], dim=1)
                # tip_p = finger_p[:, 4]
                # joint_p = finger_p[:, :4]
                # loss_t += torch.sum(torch.sum(torch.square(tip_p-tip_target[:, part_i, finger_i]), dim=1) * loss_coef[:, part_i, finger_i])
                # loss_j += torch.sum(torch.sum(torch.square(joint_p-joint_target[:, part_i, finger_i]), dim=(1,2)) * loss_coef[:, part_i, finger_i])
        pose_smooth = torch.sum(torch.square(pose[:-1] - pose[1:]))
        wrist_smooth = torch.sum(torch.square(trans[:-1] - trans[1:]))
        loss = 50*loss_t + loss_j + pose_smooth*0.05 + wrist_smooth*1000
        loss.backward()
        opt.step()

    trans = torch.cat([init_trans.unsqueeze(0), trans_t], dim=0)
    pose = torch.cat([init_pose.unsqueeze(0), pose_t], dim=0)
    return trans.detach(), pose.detach()


def FitSequence(
    ref_flag:torch.Tensor,
    ref:torch.Tensor,
    seq_1:torch.Tensor,
    seq_2:torch.Tensor,
    n_flag:torch.Tensor,
    stage_length:torch.Tensor,
    init_trans,
    init_pose,
    objs,
    obj_traj:torch.Tensor
):
    '''
    ref_flag: stage, part, finger
    ref: stage, part, finger, position + normal
    seq_1: stage, part, frame, finger, joints position
    seq_2: stage, part, frame, finger, joints position
    n_flag: stage, part, frame, finger
    objs: part, trimesh_obj
    obj_traj: stage, part, frame, 6D traj
    '''

    stage_cnt = seq_1.shape[0]
    part_cnt = seq_1.shape[1]
    len_frame = seq_1.shape[2]
    tip_target_1 = torch.zeros((stage_cnt, part_cnt, len_frame, 5, 3), dtype=torch.float32).cuda()
    joint_target_1 = torch.zeros((stage_cnt, part_cnt, len_frame, 5, 4, 3), dtype=torch.float32).cuda()
    loss_coef_1 = torch.zeros((stage_cnt, part_cnt, len_frame, 5), dtype=torch.float32).cuda()
    tip_target_2 = torch.zeros((stage_cnt, part_cnt, len_frame, 5, 3), dtype=torch.float32).cuda()
    joint_target_2 = torch.zeros((stage_cnt, part_cnt, len_frame, 5, 4, 3), dtype=torch.float32).cuda()
    loss_coef_2 = torch.zeros((stage_cnt, part_cnt, len_frame, 5), dtype=torch.float32).cuda()

    for stage_i in range(seq_1.shape[0]):
        (
          tip_target_1[stage_i],
          joint_target_1[stage_i],
          loss_coef_1[stage_i],
          tip_target_2[stage_i],
          joint_target_2[stage_i],
          loss_coef_2[stage_i]
        ) = get_stage_fitting_target(
            ref_flag[stage_i-1] if stage_i != 0 else torch.zeros_like(ref_flag[stage_i]),
            ref_flag[stage_i],
            ref[stage_i-1] if stage_i != 0 else torch.zeros_like(ref[stage_i]),
            ref[stage_i],
            seq_1[stage_i],
            seq_2[stage_i],
            n_flag[stage_i],
            stage_length[stage_i],
            objs,
            obj_traj[stage_i]
        )

    # Concatenate stages, combine results from different ends
    tip_target_1_s = []
    tip_target_2_s = []
    joint_target_1_s = []
    joint_target_2_s = []
    loss_coef_1_s = []
    loss_coef_2_s = []
    for stage_i in range(seq_1.shape[0]):
        tip_target_1_s.append(tip_target_1[stage_i, :, :stage_length[stage_i]].transpose(0, 1))
        tip_target_2_s.append(tip_target_2[stage_i, :, :stage_length[stage_i]].transpose(0, 1))
        joint_target_1_s.append(joint_target_1[stage_i, :, :stage_length[stage_i]].transpose(0, 1))
        joint_target_2_s.append(joint_target_2[stage_i, :, :stage_length[stage_i]].transpose(0, 1))
        loss_coef_1_s.append(loss_coef_1[stage_i, :, :stage_length[stage_i]].transpose(0, 1))
        loss_coef_2_s.append(loss_coef_2[stage_i, :, :stage_length[stage_i]].transpose(0, 1))
    tip_target_1 = torch.cat(tip_target_1_s, dim=0)
    tip_target_2 = torch.cat(tip_target_2_s, dim=0)
    joint_target_1 = torch.cat(joint_target_1_s, dim=0)
    joint_target_2 = torch.cat(joint_target_2_s, dim=0)
    loss_coef_1 = torch.cat(loss_coef_1_s, dim=0)
    loss_coef_2 = torch.cat(loss_coef_2_s, dim=0)
    tip_target = tip_target_1 * loss_coef_1.unsqueeze(-1) + tip_target_2 * loss_coef_2.unsqueeze(-1)
    joint_target = joint_target_1 * loss_coef_1.unsqueeze(-1).unsqueeze(-1) + joint_target_2 * loss_coef_2.unsqueeze(-1).unsqueeze(-1)
    joint_target = torch.nn.functional.normalize(joint_target, dim=4)
    loss_coef = loss_coef_1 + loss_coef_2

    # Optimization
    trans, pose = fit_joint(
        tip_target,
        joint_target,
        loss_coef,
        init_trans,
        init_pose,
    )

    return trans, pose
