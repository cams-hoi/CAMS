import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import pickle
import torch
import numpy as np
import trimesh
from manotorch.manolayer import ManoLayer
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import random
from pysdf import SDF
import copy

manolayer = ManoLayer(
    mano_assets_root='../mano_assets/',
    side='right'
)

def calc_contact_ref(obj_pts, mano_mesh, finger, mano_parts, stage, obj_part):
    tip_joint = [0, 15, 3, 6, 12, 9][finger]
    joints = [tip_joint, tip_joint-1, tip_joint-2]
    for joint_idx in joints:
        sub_mesh = trimesh.Trimesh(mano_mesh.vertices,
            mano_mesh.faces[mano_parts==joint_idx],
            process=False)
        sdf = SDF(sub_mesh.vertices, sub_mesh.faces)
        d = -sdf(obj_pts[:, :3])
        if not (d<0.01).any():
            continue
        contact_pts = obj_pts[d < 0.01, :3]

        cdist = torch.cdist(obj_pts[:, :3], contact_pts)
        ref_idx = cdist.sum(dim=1).argmin()

        if joint_idx != joints[0]:
            print('skip tip')
            import pdb; pdb.set_trace()
        # mano_projector.plot_pc(obj_pts[ref_idx:ref_idx+1, :3], mesh=mano_mesh)
        return obj_pts[ref_idx]
    return None

def convert_to_obj_frame(pc, obj_rot, obj_trans):
    pc = (obj_rot.T @ (pc - obj_trans).T).T
    return pc

# all in obj frame
mano_parts = torch.load('mano_parts_1.pt')

meta_file = '../meta/pliers_meta.torch'
all_meta = torch.load(meta_file)

results = {}
for seq_path in all_meta.keys():

    meta = all_meta[seq_path]
    length = meta['data']['length']

    hand_trans = meta['data']['hand_trans']
    hand_theta = meta['data']['hand_theta']
    mano_beta = meta['data']['beta_mano']

    stage_partition = meta['cams']['stage_partition']

    output = {}
    output['stage_partition'] = meta['cams']['stage_partition']
    output['obj'] = {}
    for part in range(2):

        obj_mesh = copy.deepcopy(meta['data']['obj'][part]['obj_mesh'])
        obj_vertices = torch.tensor(obj_mesh.vertices).float()
        obj_normals = torch.tensor(obj_mesh.vertex_normals).float()

        obj_vertices = torch.cat([obj_vertices, obj_normals], dim=-1)
        pts_range = torch.tensor([[obj_vertices[:, i].min(), obj_vertices[:, i].max()] for i in range(3)])

        obj_sdf = SDF(obj_mesh.vertices, obj_mesh.faces)

        mano_joints = []
        min_d_seq = []
        for frame_idx in range(length):
            mano_output = manolayer(hand_theta[frame_idx].unsqueeze(0), mano_beta.unsqueeze(0))
            mano_joint = mano_output.joints - mano_output.joints[0, 0] + hand_trans[frame_idx]
            mano_joint = convert_to_obj_frame(mano_joint.squeeze(),
                                            meta['data']['obj'][part]['obj_rot'][frame_idx],
                                            meta['data']['obj'][part]['obj_trans'][frame_idx])
            mano_joints.append(mano_joint)

            mano_verts = mano_output.verts[0] - mano_output.joints[0, 0] + hand_trans[frame_idx]
            mano_verts = convert_to_obj_frame(mano_verts.squeeze(),
                                            meta['data']['obj'][part]['obj_rot'][frame_idx],
                                            meta['data']['obj'][part]['obj_trans'][frame_idx])
            mano_verts_d = obj_sdf(mano_verts)
            mano_verts_d = -torch.tensor(mano_verts_d)

            min_d = torch.ones((5))
            for finger in range(5):
                tip_joint = [15, 3, 6, 12, 9][finger]
                joints = [tip_joint, tip_joint-1, tip_joint-2]
                idx = torch.any(torch.stack([mano_parts==j for j in joints]), dim=0)
                finger_verts = manolayer.get_mano_closed_faces()[idx].reshape((-1))
                min_d[finger] = mano_verts_d[finger_verts].min(dim=0)[0]
            min_d_seq.append(min_d)

        min_d_seq = torch.stack(min_d_seq, dim=0)
        mano_joints = torch.stack(mano_joints, dim=0)

        stage_contact_ref = []
        stage_normalized_contact_ref = []
        for stage in range(len(stage_partition)):
            frame_idx = stage_partition[stage]

            mano_output = manolayer(hand_theta[frame_idx].unsqueeze(0), mano_beta.unsqueeze(0))
            mano_verts = mano_output.verts[0] - mano_output.joints[0, 0] + hand_trans[frame_idx]
            mano_verts = convert_to_obj_frame(mano_verts.squeeze(),
                                            meta['data']['obj'][part]['obj_rot'][frame_idx],
                                            meta['data']['obj'][part]['obj_trans'][frame_idx])
            hand_mesh_transformed = trimesh.Trimesh(mano_verts.detach(), manolayer.get_mano_closed_faces(), process=False)

            contact_ref = [None]
            normalized_contact_ref = torch.zeros((5, 7))
            for i in range(1, 6):
                if part == 0 and i == 2:
                    contact_ref.append(None)
                    continue
                contact_ref.append(calc_contact_ref(obj_vertices, hand_mesh_transformed, i, mano_parts, stage, part))
                if contact_ref[-1] is not None:
                    normalized_contact_ref[i-1, 0] = 1
                    normalized_contact_ref[i-1, 1:4] = (contact_ref[-1][:3] - pts_range[:, 0]) \
                        / (pts_range[:, 1] - pts_range[:, 0])
                    normalized_contact_ref[i-1, 4:] = contact_ref[-1][3:]
            stage_contact_ref.append(contact_ref)
            stage_normalized_contact_ref.append(normalized_contact_ref)

        stage_joint_seq = []
        for stage in range(len(stage_partition)):
            joint_seq = torch.zeros((length, 5, 15))
            contact_ref = stage_contact_ref[stage]
            for i in range(length):
                for j in range(1, 6):
                    if contact_ref[j] is not None:
                        finger_joints = mano_joints[i, [0, j*4-3, j*4-2, j*4-1, j*4]] - contact_ref[j][:3]
                        finger_joints[:4] -= finger_joints[4]
                        finger_joints[0] /= torch.norm(finger_joints[0])
                        finger_joints[1] /= torch.norm(finger_joints[1])
                        finger_joints[2] /= torch.norm(finger_joints[2])
                        finger_joints[3] /= torch.norm(finger_joints[3])
                        joint_seq[i, j-1] = finger_joints.view(-1)
            stage_joint_seq.append(joint_seq)


        output['obj'][part] = {}
        output['obj'][part]['contact_ref'] = stage_contact_ref
        output['obj'][part]['normalized_contact_ref'] = stage_normalized_contact_ref
        output['obj'][part]['joint_seq'] = stage_joint_seq
        output['obj'][part]['pts_range'] = pts_range
        output['obj'][part]['min_dist_seq'] = min_d_seq

    results[seq_path] = {}
    results[seq_path]['data'] = meta['data']
    results[seq_path]['cams'] = output

    print(seq_path, 'OK')

torch.save(results, 'pliers_meta.torch')
