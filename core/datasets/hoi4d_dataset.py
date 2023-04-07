import os
import time
import copy
import random
import torch
import pickle
import numpy as np
import trimesh
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

from easydict import EasyDict as edict
from torch.utils.data import Dataset

class HOI4D(Dataset):

    def __init__(self, data_config):

        self.config = edict(data_config.data)
        # self.config.gpu = data_config.device
        with open(self.config.path.train, 'r') as f:
            info_list = f.readlines()

        self.meta_root = self.config.path.meta_root

        self.seq_len = self.config.seq_len
        self.n_parts = self.config.n_parts
        self.n_stages = self.config.n_stages
        self.data_info = []
        for info in info_list:

            seq_path = info.strip()
            self.data_info.append(seq_path)

        self.meta = torch.load(self.meta_root)

    def prepare_data(self, idx):

        seq_path = self.data_info[idx]

        obj_vertices = []
        for part in range(self.n_parts):
            obj_mesh = self.meta[seq_path]['data']['obj'][part]['obj_mesh']
            can_obj_vertices = torch.tensor(obj_mesh.vertices).float()

            init_pts_num = can_obj_vertices.shape[0]
            sample_index = random.sample(range(init_pts_num), k=self.config.sample_num)
            sample_index = torch.tensor(sample_index).long()
            can_obj_vertices = can_obj_vertices[sample_index]
            can_obj_vertices = self.augment(can_obj_vertices, self.meta[seq_path]['cams']['obj'][part]['pts_range'])
            obj_vertices.append(can_obj_vertices)
        obj_vertices = torch.stack(obj_vertices)

        contact_ref = torch.zeros((self.n_stages, self.n_parts, 5, 7))
        control = torch.zeros((self.n_parts, 51))
        pts_range = torch.zeros((self.n_parts, 3, 2))

        t_seq = torch.rand((self.n_stages, self.n_parts, self.seq_len))
        in_seq_1 = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5, 15))
        in_seq_2 = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5, 15))
        out_seq_1 = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5, 15))
        out_seq_2 = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5, 15))

        in_contact_flag_seq = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5))
        out_contact_flag_seq = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5))
        in_near_flag_seq = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5))
        out_near_flag_seq = torch.zeros((self.n_stages, self.n_parts, self.seq_len, 5))

        for part in range(self.n_parts):
            pts_range[part] = self.meta[seq_path]['cams']['obj'][part]['pts_range']

            for stage in range(self.n_stages):
                contact_ref[stage, part] = self.meta[seq_path]['cams']['obj'][part]['normalized_contact_ref'][stage+1]

                start = self.meta[seq_path]['cams']['stage_partition'][stage]
                end = self.meta[seq_path]['cams']['stage_partition'][stage+1]

                in_frames = torch.linspace(start, end, self.seq_len, dtype=torch.long)
                in_seq_1[stage, part] = self.meta[seq_path]['cams']['obj'][part]['joint_seq'][stage][in_frames].view(self.seq_len, 5, 15)
                in_seq_2[stage, part] = self.meta[seq_path]['cams']['obj'][part]['joint_seq'][stage+1][in_frames].view(self.seq_len, 5, 15)
                in_contact_flag_seq[stage, part] = self.meta[seq_path]['cams']['obj'][part]['min_dist_seq'][in_frames] < 0.01
                in_near_flag_seq[stage, part] = self.meta[seq_path]['cams']['obj'][part]['min_dist_seq'][in_frames] < 0.1

                out_frames = (t_seq[stage, part] * (end-start) + start).long()
                out_seq_1[stage, part] = self.meta[seq_path]['cams']['obj'][part]['joint_seq'][stage][out_frames].view(self.seq_len, 5, 15)
                out_seq_2[stage, part] = self.meta[seq_path]['cams']['obj'][part]['joint_seq'][stage+1][out_frames].view(self.seq_len, 5, 15)
                out_contact_flag_seq[stage, part] = self.meta[seq_path]['cams']['obj'][part]['min_dist_seq'][out_frames] < 0.01
                out_near_flag_seq[stage, part] = self.meta[seq_path]['cams']['obj'][part]['min_dist_seq'][out_frames] < 0.1

            hand_theta_world = self.meta[seq_path]['data']['hand_theta'][0]
            hand_trans_world = self.meta[seq_path]['data']['hand_trans'][0]
            obj_rot_world = self.meta[seq_path]['data']['obj'][part]['obj_rot'][0]
            obj_trans_world = self.meta[seq_path]['data']['obj'][part]['obj_trans'][0]

            hand_wrist_rot_world = torch.tensor(R.from_rotvec(hand_theta_world[:3]).as_matrix(), dtype=torch.float32)
            hand_wrist_rot_to_part = torch.matmul(obj_rot_world.transpose(0, 1), hand_wrist_rot_world)
            hand_wrist_trans_world = hand_trans_world
            hand_wrist_trans_to_part = torch.matmul(obj_rot_world.transpose(0, 1), hand_wrist_trans_world.unsqueeze(-1) - obj_trans_world.unsqueeze(-1)).squeeze()

            hand_theta_to_part = hand_theta_world
            hand_theta_to_part[:3] = torch.tensor(R.from_matrix(hand_wrist_rot_to_part).as_rotvec(), dtype=torch.float32)
            control[part, :48] = hand_theta_to_part
            control[part, 48:] = hand_wrist_trans_to_part

        control = control.reshape(-1)

        stage_partition = self.meta[seq_path]['cams']['stage_partition']
        stage_length = []
        for i in range(len(stage_partition)-1):
            stage_length.append(stage_partition[i+1] - stage_partition[i])
        if len(stage_length) == 3:
            stage_length[1] -= 1
            stage_length[2] += 2
        elif len(stage_length) == 2:
            stage_length[1] += 1
        stage_length = torch.tensor(stage_length).long()

        sample = {'obj_vertices': obj_vertices,
                'contact_ref': contact_ref,
                'in_seq_1': in_seq_1,
                'in_seq_2': in_seq_2,
                'in_contact_flag_seq': in_contact_flag_seq,
                'in_near_flag_seq': in_near_flag_seq,
                'out_seq_1': out_seq_1,
                'out_seq_2': out_seq_2,
                'out_contact_flag_seq': out_contact_flag_seq,
                'out_near_flag_seq': out_near_flag_seq,
                'stage_length': stage_length,
                't_seq': t_seq,
                'control': control,
                'pts_range': pts_range,
                'mano_beta': self.meta[seq_path]['data']['beta_mano'],
                'seq_path': seq_path}

        return sample

    def augment(self, obj_vertices, pts_range):

        pts_range = pts_range.transpose(0, 1)

        # obj_vertices += torch.randn(obj_vertices.shape) * 0.005
        obj_vertices = (obj_vertices - pts_range[0]) / (pts_range[1] - pts_range[0])

        return obj_vertices

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):

        sample = self.prepare_data(idx)

        return sample

    def get_test_dataset(self):
        return TestDataset(self.config)

class TestDataset(HOI4D):

    def __init__(self, data_config):

        self.config = data_config
        with open(self.config.path.test, 'r') as f:
            info_list = f.readlines()

        self.meta_root = self.config.path.meta_root

        self.seq_len = self.config.seq_len
        self.n_parts = self.config.n_parts
        self.n_stages = self.config.n_stages
        self.data_info = []
        for info in info_list:

            seq_path = info.strip()
            self.data_info.append(seq_path)

        self.meta = torch.load(self.meta_root)
