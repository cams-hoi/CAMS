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
import open3d as o3d
from tqdm import tqdm
from manotorch import manolayer
from manotorch import axislayer
from sklearn.neighbors import KDTree
from pysdf import SDF
import argparse

from core.utils.amano import ManoLayer as AManoLayer

mano_layer = AManoLayer()

def get_inertia(obj):
    I = torch.zeros((3, 3), dtype=torch.float32)
    com = torch.mean(torch.from_numpy(obj.vertices), dim=0)
    verts = torch.from_numpy(obj.vertices) - com
    for i in range(3):
        for j in range(3):
            if i==j:
                I[i, j] = torch.sum(torch.square(verts[:,(i+1)%3]) + torch.square(verts[:,(i+2)%3]))
            else:
                I[i, j] = -torch.sum(verts[:, i] * verts[:, j])
    return I

def calculate_fix(frame1, frame2):
    mat1_0, mat1_1 = torch.eye(4), torch.eye(4)
    mat2_0, mat2_1 = torch.eye(4), torch.eye(4)
    mat1_0[:3, :3], mat1_0[:3, 3] = torch.from_numpy(R.from_rotvec(frame1[0, 3:]).as_matrix()), frame1[0, :3]
    mat1_1[:3, :3], mat1_1[:3, 3] = torch.from_numpy(R.from_rotvec(frame1[1, 3:]).as_matrix()), frame1[1, :3]
    mat2_0[:3, :3], mat2_0[:3, 3] = torch.from_numpy(R.from_rotvec(frame2[0, 3:]).as_matrix()), frame2[0, :3]
    mat2_1[:3, :3], mat2_1[:3, 3] = torch.from_numpy(R.from_rotvec(frame2[1, 3:]).as_matrix()), frame2[1, :3]

    mat1 = torch.matmul(torch.linalg.inv(mat1_0), mat1_1)
    mat2 = torch.matmul(torch.linalg.inv(mat2_0), mat2_1)

    d_mat = mat2.matmul(torch.inverse(mat1))
    L, V = torch.linalg.eig(d_mat)
    fix = None
    for i in range(4):
        if abs(L[i]-1)<1e-5 and abs(V[3, i])>1e-5:
            fix = (V[:3, i] / V[3, i]).real
    if fix is None:
        print('part interpolate error')

    rot_vec = torch.tensor(R.from_matrix(d_mat[:3, :3]).as_rotvec()).float()
    return fix, rot_vec

def get_hand_seq(trans, pose):
    mano_output = mano_layer(pose[:, :3], pose[:, 3:])
    mano_verts = (mano_output.verts - mano_output.joints[:, :1] + trans.unsqueeze(1)).detach()
    return mano_verts

def get_pen_depth(trans, pose, objs, obj_traj):
    max_u = 0
    hand_verts = get_hand_seq(trans, pose)
    for (obj_i, obj) in enumerate(objs):
        traj = obj_traj[:, obj_i, :]
        sdf = SDF(obj.vertices, obj.faces)
        for frame_i in range(traj.shape[0]):
            cur_verts = R.from_rotvec(traj[frame_i, 3:]).inv().apply(hand_verts[frame_i] - traj[frame_i, :3])
            u = np.where(sdf(cur_verts) > 0.005)[0]
            max_u = max_u + u.shape[0]
    return max_u / trans.shape[0] / 778

from scipy.optimize import nnls
class LinearSystem():
    def __init__(self, coms):
        self.b = torch.zeros(12, dtype=torch.float32)
        self.A = []
        self.coms = coms

    def add_constant(self, id, f, t):
        if id == 0:
            self.b[:6] += torch.cat([f, t], dim=0)
        else:
            self.b[6:] += torch.cat([f, t], dim=0)

    def add_constraint(self, id, f, c):
        if id == 0:
            a = torch.cat([f, torch.cross(c - self.coms[0], f), torch.zeros(6)], dim=0)
        else:
            a = torch.cat([torch.zeros(6), f, torch.cross(c - self.coms[1], f)], dim=0)
        self.A.append(a)

    def add_both_constraint(self, f1, c1, f2, c2):
        a = torch.cat([f1, torch.cross(c1 - self.coms[0], f1), f2, torch.cross(c2 - self.coms[1], f2)], dim=0)
        self.A.append(a)

    def optimize(self):
        if len(self.A) == 0:
            self.A.append(torch.zeros(12))
        A = torch.stack(self.A, dim=0).transpose(0,1)
        x = nnls(A, self.b)
        # print(x)
        return x[1] < 0.01

def get_phy_score(trans, pose, objs, obj_traj):
    phy_score = np.zeros(obj_traj.shape[0], dtype = np.int32)
    hand_verts = get_hand_seq(trans, pose)
    com_traj = torch.zeros((obj_traj.shape[0], obj_traj.shape[1], 4, 4), dtype=torch.float32)
    com_traj[:, :, 3, 3] = 1
    for (obj_i, obj) in enumerate(objs):
        com = torch.mean(torch.from_numpy(obj.vertices), dim=0)
        com_traj[:, obj_i, :3, 3] = torch.from_numpy(R.from_rotvec(obj_traj[:, obj_i, 3:]).apply(com)) + obj_traj[:, obj_i, :3]
        com_traj[:, obj_i, :3, :3] = torch.from_numpy(R.from_rotvec(obj_traj[:, obj_i, 3:]).as_matrix())
    v = torch.zeros((obj_traj.shape[0], obj_traj.shape[1], 3), dtype=torch.float32)
    w = torch.zeros((obj_traj.shape[0], obj_traj.shape[1], 3), dtype=torch.float32)
    for (obj_i, obj) in enumerate(objs):
        for frame_i in range(obj_traj.shape[0]-1):
            transmat = torch.matmul(torch.linalg.inv(com_traj[frame_i, obj_i]), com_traj[frame_i+1, obj_i])
            v[frame_i, obj_i] = transmat[:3, 3]
            w[frame_i, obj_i] = torch.from_numpy(R.from_matrix(transmat[:3, :3]).as_rotvec())
    v_ = torch.zeros_like(v)
    w_ = torch.zeros_like(w)
    v_[0:-2] = v[1:-1] - v[0:-2]
    w_[0:-2] = w[1:-1] - w[0:-2]
    P = torch.zeros_like(v)
    P_ = torch.zeros_like(v_)
    for (obj_i, obj) in enumerate(objs):
        P[:, obj_i] = v[:, obj_i] * obj.vertices.shape[0] # Assume each vert has 1 mass
        P_[:, obj_i] = v_[:, obj_i] * obj.vertices.shape[0]
    L = torch.zeros((obj_traj.shape[0], 2, 3), dtype=torch.float32)
    L_ = torch.zeros((obj_traj.shape[0], 2, 3), dtype=torch.float32)
    for (obj_i, obj) in enumerate(objs):
        I0 = get_inertia(obj)
        for frame_i in range(obj_traj.shape[0]-2):
            Rt = com_traj[frame_i, obj_i, :3, :3]
            wt = w[frame_i, obj_i]
            wt_ = w_[frame_i, obj_i]
            skew = torch.Tensor([[0, -wt[2], wt[1]], [wt[2], 0, -wt[0]], [-wt[1], wt[0], 0]])
            R_ = torch.matmul(skew, Rt).transpose(0,1)
            I = torch.matmul(torch.matmul(Rt, I0), Rt.transpose(0,1))
            I_ = torch.matmul(torch.matmul(R_, I0), Rt.transpose(0,1)) + torch.matmul(torch.matmul(Rt, I0), R_.transpose(0,1))
            L[frame_i, obj_i] = torch.matmul(I, wt.unsqueeze(-1)).squeeze()
            L_[frame_i, obj_i] = torch.matmul(I_, wt.unsqueeze(-1)).squeeze() + torch.matmul(I, wt_.unsqueeze(-1)).squeeze()
    sdf = []
    for (obj_i, obj) in enumerate(objs):
        sdf.append(SDF(obj.vertices, obj.faces))
    table_top = 100
    for (obj_i, obj) in enumerate(objs):
        cur_verts = torch.from_numpy(R.from_rotvec(obj_traj[0, obj_i, 3:]).apply(obj.vertices)) + obj_traj[0, obj_i, :3]
        table_top = min(table_top, torch.min(cur_verts[:,2]))
    if len(objs) == 2:
        fix, axis = calculate_fix(obj_traj[0], obj_traj[-1])
        axis = axis / torch.norm(axis)
        verts = torch.from_numpy(objs[0].vertices)
        para = torch.sum((verts - fix) * axis, dim=1)
        p_min, p_max = torch.min(para), torch.max(para)
        axis_1, axis_2 = fix + p_min * axis, fix + p_max * axis

    for frame_i in range(1, obj_traj.shape[0]-1):
        Pt, Lt = P_[frame_i-1], L_[frame_i-1]
        h_verts = hand_verts[frame_i]
        coms = torch.zeros((2, 3), dtype=torch.float32)
        for (obj_i, obj) in enumerate(objs):
            cur_verts = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).apply(obj.vertices)) + obj_traj[frame_i, obj_i, :3]
            coms[obj_i] = torch.mean(cur_verts, dim=0)
        AX = LinearSystem(coms)
        for (obj_i, obj) in enumerate(objs):
            AX.add_constant(obj_i, Pt[obj_i], Lt[obj_i])
            AX.add_constant(obj_i, -obj.vertices.shape[0]*torch.tensor([0,0,-1]), torch.zeros(3))
            cur_h = R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).inv().apply(h_verts - obj_traj[frame_i, obj_i, :3])
            dist, nn = sdf[obj_i](cur_h), sdf[obj_i].nn(cur_h)
            contact_idx = torch.from_numpy(nn[np.where(dist > -0.002)])
            cur_verts = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).apply(obj.vertices)) + obj_traj[frame_i, obj_i, :3]
            cur_norms = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).apply(obj.vertex_normals))
            table_top_idx = torch.where(cur_verts[:,2] < table_top + 0.005)[0]
            if table_top_idx.shape[0] > 0:
                samples = torch.randint(0, table_top_idx.shape[0], [20])
                table_top_idx = table_top_idx[samples]
            for idx_i in range(contact_idx.shape[0]):
                idx = contact_idx[idx_i]
                c = cur_verts[idx].float()
                n = cur_norms[idx].float()
                n1 = torch.cross(n, torch.rand(3, dtype=torch.float32))
                n1 = n1 / n1.norm()
                n2 = torch.cross(n, n1)
                AX.add_constraint(obj_i, -n+0.35*n1, c)
                AX.add_constraint(obj_i, -n-0.35*n1, c)
                AX.add_constraint(obj_i, -n+0.35*n2, c)
                AX.add_constraint(obj_i, -n-0.35*n2, c)
            for idx_i in range(table_top_idx.shape[0]):
                idx = table_top_idx[idx_i]
                c = cur_verts[idx].float()
                n = torch.tensor([0, 0, -1])
                n1 = torch.tensor([0, 1, 0])
                n2 = torch.tensor([1, 0, 0])
                AX.add_constraint(obj_i, -n+0.35*n1, c)
                AX.add_constraint(obj_i, -n-0.35*n1, c)
                AX.add_constraint(obj_i, -n+0.35*n2, c)
                AX.add_constraint(obj_i, -n-0.35*n2, c)
        if len(objs) == 2:
            axis_1_p = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, 0, 3:]).apply(axis_1)).float() + obj_traj[frame_i, 0, :3]
            axis_2_p = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, 0, 3:]).apply(axis_2)).float() + obj_traj[frame_i, 0, :3]

            for u in range(0, 11):
                c = (axis_1_p * u + axis_2_p * (10 - u)) / 10
                n = axis_2_p - axis_1_p
                n1 = torch.cross(n, torch.rand(3, dtype=torch.float32))
                n2 = torch.cross(n, n1)
                n1 = n1 / n1.norm()
                n2 = n2 / n2.norm()
                AX.add_both_constraint(n1, c, -n1, c)
                AX.add_both_constraint(-n1, c, n1, c)
                AX.add_both_constraint(n2, c, -n2, c)
                AX.add_both_constraint(-n2, c, n2, c)
        phy_score[frame_i] = AX.optimize()
    # print(phy_score)
    return phy_score.sum() / (obj_traj.shape[0]-2)

def get_art_score(trans, pose, objs, obj_traj):
    hand_verts = get_hand_seq(trans, pose)
    if len(objs) == 1:
        return 1.0
    fix, axis = calculate_fix(obj_traj[0], obj_traj[-1])
    axis = axis / torch.norm(axis)
    verts = torch.from_numpy(objs[0].vertices)
    para = torch.sum((verts - fix) * axis, dim=1)
    p_min, p_max = torch.min(para), torch.max(para)
    axis_1, axis_2 = fix + p_min * axis, fix + p_max * axis
    tot_frame, pass_frame = 0, 0
    sdf = []
    for (obj_i, obj) in enumerate(objs):
        sdf.append(SDF(obj.vertices, obj.faces))
    table_top = 100
    for (obj_i, obj) in enumerate(objs):
        cur_verts = torch.from_numpy(R.from_rotvec(obj_traj[0, obj_i, 3:]).apply(obj.vertices)) + obj_traj[0, obj_i, :3]
        table_top = min(table_top, torch.min(cur_verts[:,2]))
    for frame_i in range(obj_traj.shape[0] - 1):
        mat1_0, mat1_1 = torch.eye(4), torch.eye(4)
        mat2_0, mat2_1 = torch.eye(4), torch.eye(4)
        mat1_0[:3, :3], mat1_0[:3, 3] = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, 0, 3:]).as_matrix()), obj_traj[frame_i, 0, :3]
        mat1_1[:3, :3], mat1_1[:3, 3] = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, 1, 3:]).as_matrix()), obj_traj[frame_i, 1, :3]
        mat2_0[:3, :3], mat2_0[:3, 3] = torch.from_numpy(R.from_rotvec(obj_traj[frame_i+1, 0, 3:]).as_matrix()), obj_traj[frame_i+1, 0, :3]
        mat2_1[:3, :3], mat2_1[:3, 3] = torch.from_numpy(R.from_rotvec(obj_traj[frame_i+1, 1, 3:]).as_matrix()), obj_traj[frame_i+1, 1, :3]
        mat1 = torch.matmul(mat1_1, torch.linalg.inv(mat1_0))
        mat2 = torch.matmul(mat2_1, torch.linalg.inv(mat2_0))
        d_mat = mat2.matmul(torch.inverse(mat1))
        rot_vec = torch.from_numpy(R.from_matrix(d_mat[:3, :3]).as_rotvec())
        if torch.norm(rot_vec) > 1e-6:
            tot_frame += 1
            axis_1_p = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, 0, 3:]).apply(axis_1)).float() + obj_traj[frame_i, 0, :3]
            axis_2_p = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, 0, 3:]).apply(axis_2)).float() + obj_traj[frame_i, 0, :3]
            u0, u1 = 0, 0
            for (obj_i, obj) in enumerate(objs):
                axis = axis_2_p - axis_1_p
                axis = axis / torch.norm(axis)
                max_torque = 0
                min_torque = 0
                cur_h = R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).inv().apply(hand_verts[frame_i] - obj_traj[frame_i, obj_i, :3])
                dist, nn = sdf[obj_i](cur_h), sdf[obj_i].nn(cur_h)
                contact_idx = torch.from_numpy(nn[np.where(dist > -0.002)])
                cur_verts = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).apply(obj.vertices)) + obj_traj[frame_i, obj_i, :3]
                cur_norms = torch.from_numpy(R.from_rotvec(obj_traj[frame_i, obj_i, 3:]).apply(obj.vertex_normals))
                table_top_idx = torch.where(cur_verts[:,2] < table_top + 0.005)[0]
                if table_top_idx.shape[0] > 0:
                    samples = torch.randint(0, table_top_idx.shape[0], [50])
                    table_top_idx = table_top_idx[samples]
                tmp = (cur_verts - axis_1_p) - torch.sum((cur_verts - axis_1_p) * axis, dim=1).unsqueeze(-1).repeat(1, 3) * axis
                avg_d = torch.mean(torch.sqrt(torch.sum(torch.square(tmp), dim=1)))
                def update(c, n):
                    c = (c - axis_1_p) - torch.sum((c - axis_1_p) * axis) * axis
                    n = n - torch.sum(n * axis) * axis
                    tor = torch.cross(n, c)
                    return torch.sum(tor * axis)
                for idx_i in range(contact_idx.shape[0]):
                    idx = contact_idx[idx_i]
                    g = update(cur_verts[idx].float(), -cur_norms[idx].float())
                    max_torque,min_torque=max(max_torque,g), min(min_torque,g)
                g = update(torch.mean(cur_verts, dim=0).float(), torch.tensor((0, 0, -1), dtype=torch.float32))
                max_torque,min_torque=max(max_torque,g), min(min_torque,g)
                for idx_i in range(table_top_idx.shape[0]):
                    idx = table_top_idx[idx_i]
                    g = update(cur_verts[idx].float(), torch.tensor([0, 0, 1]))
                    max_torque,min_torque=max(max_torque,g), min(min_torque,g)
                if obj_i == 0:
                    u0 = max_torque / avg_d
                else:
                    u1 = -min_torque / avg_d
            # print(u0, u1)
            if u0 > 0.3 and u1 > 0.3:
                pass_frame += 1
    return pass_frame / tot_frame

'''
"hand_trans": N * 3
"hand_pose": N * 48
"objs": [obj1, obj2, ...] trimesh
"obj_traj": N * part * 6, [0:3] is transition, [3:6] is rotvec
'''

import math,os
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Synthesizer")

    parser.add_argument('--file_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')

    args = parser.parse_args()
    root_path = args.file_path
    save_path = args.save_path

    dirs = os.listdir(root_path)
    p1, p2, p3 = 0, 0, 0
    cnt = 0
    for dir in dirs:

        dir_path = os.path.join(root_path, dir)
        files = os.listdir(dir_path)
        cnt += len(files)
        for file in files:

            file_path = os.path.join(dir_path, file)
            data = torch.load(file_path)
            trans = data["hand_trans"].detach().cpu()
            pose = data["hand_pose"].detach().cpu()
            objs = data["objs"]
            obj_traj = data["obj_traj"].detach().cpu()

            penetration = get_pen_depth(trans, pose, objs, obj_traj)
            print("MAX PEN POR: {}".format(penetration))

            physics_score = get_phy_score(trans, pose, objs, obj_traj)
            print("PHYSICS SCORE: {}".format(physics_score))

            articulation_score = get_art_score(trans, pose, objs, obj_traj)
            print("ARTICULATION SCORE: {}".format(articulation_score))

            p1 += penetration
            p2 += physics_score
            p3 += articulation_score

    lines = ['{} {} {}\n'.format(p1/cnt,p2/cnt,p3/cnt)]
    with open(save_path, 'w') as f:
        f.writelines(lines)
