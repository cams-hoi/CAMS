import numpy as np
import trimesh
import trimesh.sample
import trimesh.geometry
import trimesh.ray
import trimesh.proximity
import matplotlib.pyplot as plt
from manotorch.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import json
import pickle
import sys
from tqdm import tqdm
from manotorch import manolayer
from manotorch import axislayer
from sklearn.neighbors import KDTree
from pysdf import SDF
import argparse

from synthesizer.FitSequence import FitSequence
from synthesizer.FitContact import FitContact

def Synthesis(
    ref_flag:torch.Tensor,
    ref:torch.Tensor,
    seq_1:torch.Tensor,
    seq_2:torch.Tensor,
    c_flag:torch.Tensor,
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
    c_flag: stage, part, frame, finger
    n_flag: stage, part, frame, finger
    objs: part, trimesh_obj
    obj_traj: stage, part, frame, 6D traj
    '''
    c_flag = c_flag.clip(0, 1)
    n_flag = n_flag.clip(0, 1)
    obj_sdf = []
    for obj in objs:
        obj_sdf.append(SDF(obj.vertices, obj.faces))

    # Fit an initial pose according to CAMS
    trans, pose = FitSequence(ref_flag, ref, seq_1, seq_2,
        n_flag, stage_length, init_trans, init_pose, objs, obj_traj)

    # Fit contact according to the contact information
    init_trans = trans.clone().detach()
    for iter in tqdm(range(6)):
        smooth_coef = [1, 1, 10, 10, 500, 500][iter]
        trans, pose = FitContact(trans, init_trans, pose, ref_flag, ref,
            c_flag, stage_length, objs, obj_sdf, obj_traj, smooth_coef)

    return trans, pose

import os

def test_case(output_file, meta_path, save_dir):
    output = torch.load(output_file, map_location='cpu')
    meta_data = torch.load(meta_path)

    pts_min, pts_max = output["pts_range"][:, :, :, 0].unsqueeze(2).unsqueeze(1), output["pts_range"][:, :, :, 1].unsqueeze(2).unsqueeze(1)
    output["pred_ref"][:, :, :, :, :3] = pts_min + output["pred_ref"][:, :, :, :, :3] * (pts_max - pts_min)
    for i in range(output["pred_ref"].shape[0]):
        seq_name = output['seq_path'][i]
        # if i > 0 and output['seq_path'][i] == output['seq_path'][i-1]:
        #     continue
        # meta_data = torch.load(os.path.join(meta_path, "{}.torch".format(seq_name)))
        init_trans = meta_data[seq_name]['data']['hand_trans'][0].cuda()
        init_pose = meta_data[seq_name]['data']['hand_theta'][0].cuda()
        objs = []
        obj_traj = torch.zeros((output["pred_c_flag"].shape[1], output["pred_c_flag"].shape[2], output["pred_c_flag"].shape[3], 6), dtype=torch.float32)
        pred_ref_flag = output["pred_ref_flag"][i].detach().cuda()
        pred_ref = output["pred_ref"][i].detach().cuda()
        pred_seq_1 = output["pred_seq_1"][i].detach().cuda()
        pred_seq_2 = output["pred_seq_2"][i].detach().cuda()
        pred_c_flag = output["pred_c_flag"][i].detach().cuda()
        pred_n_flag = output["pred_n_flag"][i].detach().cuda()
        stage_length = output["stage_length"][i].detach().cuda().clip(0, 100)
        obj_traj = torch.zeros((pred_c_flag.shape[0], pred_c_flag.shape[1], pred_c_flag.shape[2], 6), dtype=torch.float32)
        for part_i in range(pred_c_flag.shape[1]):
            # Generate object trajectory, using bezier curve
            objs.append(meta_data[seq_name]['data']['obj'][part_i]['obj_mesh'])
            obj_T1 = torch.zeros((len(meta_data[seq_name]['data']['obj'][part_i]['obj_trans']), 4, 4))
            obj_T2 = torch.zeros((len(meta_data[seq_name]['data']['obj'][part_i]['obj_trans']), 4, 4))
            obj_T1[:, :3, 3] = 0 if meta_data[seq_name]['data']['global_trans'] is None else meta_data[seq_name]['data']['global_trans']
            obj_T1[:, :3, :3] = torch.eye(3, 3) if meta_data[seq_name]['data']['global_rot'] is None else meta_data[seq_name]['data']['global_rot']
            obj_T1[:, 3, 3] = 1
            if isinstance(meta_data[seq_name]['data']['obj'][part_i]['obj_trans'], list):
                obj_T2[:, :3, 3] = torch.stack(meta_data[seq_name]['data']['obj'][part_i]['obj_trans'], dim=0)
                obj_T2[:, :3, :3] = torch.stack(meta_data[seq_name]['data']['obj'][part_i]['obj_rot'], dim=0)
            else:
                obj_T2[:, :3, 3] = meta_data[seq_name]['data']['obj'][part_i]['obj_trans']
                obj_T2[:, :3, :3] = meta_data[seq_name]['data']['obj'][part_i]['obj_rot']
            obj_T2[:, 3, 3] = 1
            def bezier(x1, y1, x2, y2, t):
                x0, y0 = 0, 0
                x3, y3 = 1, 1
                s0, s1 = 0, 1
                while s1-s0>1e-5:
                    s = (s0+s1)/2
                    ts = x0*(1-s)**3 + 3*x1*s*(1-s)**2 + 3*x2*(1-s)*s**2 + x3*s**3
                    if ts<t: s0 = s
                    else: s1 = s
                s = (s0+s1)/2
                return y0*(1-s)**3 + 3*y1*s*(1-s)**2 + 3*y2*(1-s)*s**2 + y3*s**3
            for stage_i in range(pred_c_flag.shape[0]):
                Lx,Rx = torch.sum(stage_length[:stage_i]), torch.sum(stage_length[:stage_i+1])
                for frame_i in range(stage_length[stage_i]):
                    w = bezier(0.4, 0, 0.6, 1, frame_i/(Rx-Lx))
                    obj_T = torch.eye(4)
                    obj_T1_R = obj_T1[Rx] if Rx < obj_T1.shape[0] else obj_T1[Rx-1]
                    obj_T1_L = obj_T1[Lx]
                    obj_T[:3, 3] = (obj_T1_R[:3, 3] - obj_T1_L[:3, 3]) * w + obj_T1_L[:3, 3]
                    delta = (R.from_matrix(obj_T1_R[:3, :3]) * R.from_matrix(obj_T1_L[:3, :3]).inv()).as_rotvec()
                    obj_T[:3, :3] = torch.from_numpy((R.from_rotvec(delta*w) * R.from_matrix(obj_T1_L[:3, :3])).as_matrix())
                    obj_T = torch.matmul(obj_T, obj_T2[Lx+frame_i])
                    obj_traj[stage_i, part_i, frame_i, :3] = obj_T[:3, 3]
                    obj_traj[stage_i, part_i, frame_i, 3:] = torch.from_numpy(R.from_matrix(obj_T[:3, :3]).as_rotvec())
        for stage_i in range(pred_c_flag.shape[0]):
            samples = torch.linspace(0, pred_c_flag.shape[2]-1, stage_length[stage_i], dtype=torch.long)
            pred_seq_1[stage_i, :, :stage_length[stage_i]] = pred_seq_1[stage_i, :, samples].clone()
            pred_seq_2[stage_i, :, :stage_length[stage_i]] = pred_seq_2[stage_i, :, samples].clone()
            pred_c_flag[stage_i, :, :stage_length[stage_i]] = pred_c_flag[stage_i, :, samples].clone()
            pred_n_flag[stage_i, :, :stage_length[stage_i]] = pred_n_flag[stage_i, :, samples].clone()

        # Synthesis
        trans, pose = Synthesis(
            pred_ref_flag,
            pred_ref,
            pred_seq_1,
            pred_seq_2,
            pred_c_flag,
            pred_n_flag,
            stage_length,
            init_trans,
            init_pose,
            objs,
            obj_traj.detach().cuda(),
        )

        # Rearrange and save
        obj_traj_s = []
        for s_i in range(obj_traj.shape[0]):
            obj_traj_s.append(obj_traj[s_i, :, :stage_length[s_i]].transpose(0,1))
        obj_traj = torch.cat(obj_traj_s, dim=0)
        result = {
            "hand_trans": trans.detach().cpu(),
            "hand_pose": pose.detach().cpu(),
            "objs": objs,
            "obj_traj": obj_traj.detach().cpu()
        }

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, '{}_{}.pt'.format(output['seq_path'][i].split('/')[-1], i))
        torch.save(result, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesizer")

    parser.add_argument('--output-file', type=str, default='')
    parser.add_argument('--meta-path', type=str, default='')

    args = parser.parse_args()
    output_file = args.output_file
    meta_path = args.meta_path

    files = os.listdir(output_file)
    for file in files:

        if file.startswith('output'):

            file_path = os.path.join(output_file, file)
            save_dir = file_path.split('/')[-1]
            save_dir = os.path.join('/'.join(file_path.split('/')[:-3]), 'synth', save_dir)

            test_case(file_path, meta_path, save_dir)
