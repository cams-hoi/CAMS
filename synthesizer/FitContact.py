import numpy as np
import torch
import trimesh
import trimesh.proximity
from scipy.spatial.transform import Rotation as R
import pickle
from tqdm import tqdm
from pysdf import SDF
from sklearn.neighbors import KDTree

from core.utils.amano import ManoLayer as AManoLayer

mano_layer = AManoLayer(cuda = True)

mano_parts_file = 'data/preparation/mano_parts.pkl'  # MANO part file of fingers
mano_knuckles_file = 'data/preparation/mano_parts_1.pt' # MANO part file of knuckles

mano_layer = AManoLayer(cuda = True)

mano_faces = mano_layer.mano_layer.th_faces.cpu()
mano_parts = pickle.load(open(mano_parts_file, 'rb'))
verts_part = np.zeros(778, dtype=np.int32)
for face_id in range(mano_faces.shape[0]):
    verts_part[mano_faces[face_id, 0]] = mano_parts[face_id]
    verts_part[mano_faces[face_id, 1]] = mano_parts[face_id]
    verts_part[mano_faces[face_id, 2]] = mano_parts[face_id]
verts_part = torch.from_numpy(verts_part).cuda()

mano_parts_d = torch.load(mano_knuckles_file)
verts_part_d = np.zeros(778, dtype=np.int32)
for face_id in range(mano_faces.shape[0]):
    verts_part_d[mano_faces[face_id, 0]] = mano_parts_d[face_id]
    verts_part_d[mano_faces[face_id, 1]] = mano_parts_d[face_id]
    verts_part_d[mano_faces[face_id, 2]] = mano_parts_d[face_id]
verts_part_d = torch.from_numpy(verts_part_d).cuda()

def FitContact(
    trans:torch.Tensor,
    init_trans:torch.Tensor,
    pose:torch.Tensor,
    ref_flag:torch.Tensor,
    ref:torch.Tensor,
    c_flag:torch.Tensor,
    stage_length:torch.Tensor,
    objs,
    obj_sdf,
    obj_traj,
    smooth_coef
):
    part_cnt = c_flag.shape[1]
    mano_output = mano_layer(pose[:, :3], pose[:, 3:])
    hand_verts = mano_output.verts - mano_output.joints[:, :1] + trans.unsqueeze(1)

    opt_hand_idx = []
    opt_target = []
    opt_coef = []

    def get_partial_mesh(obj_mesh, ref_frame, ref_normal):
        # Get the half mesh that align with the contact normal
        obj_vertex = torch.from_numpy(obj_mesh.vertices.copy()).cuda()
        obj_normal = torch.from_numpy(obj_mesh.vertex_normals.copy()).cuda()
        ref_normal = ref_normal / torch.norm(ref_normal)
        dotp = torch.sum(ref_normal * obj_normal, axis=1)
        obj_part_flag = torch.where(dotp > 0.707, True, False)
        obj_faces = torch.from_numpy(obj_mesh.faces).cuda()
        obj_part_e01 = torch.where(torch.logical_and(obj_part_flag[obj_faces[:,0]], obj_part_flag[obj_faces[:,1]]) == True)[0]
        obj_part_e02 = torch.where(torch.logical_and(obj_part_flag[obj_faces[:,0]], obj_part_flag[obj_faces[:,2]]) == True)[0]
        obj_part_e12 = torch.where(torch.logical_and(obj_part_flag[obj_faces[:,1]], obj_part_flag[obj_faces[:,2]]) == True)[0]
        obj_part_edges = torch.cat([obj_faces[obj_part_e01][:,[0,1]], obj_faces[obj_part_e02][:,[0,2]], obj_faces[obj_part_e12][:,[1,2]]], dim=0)
        obj_adj_label = torch.from_numpy(trimesh.graph.connected_component_labels(obj_part_edges.cpu(), node_count=obj_vertex.shape[0])).cuda()
        obj_part_idx = torch.where(obj_part_flag == True)[0]
        kdt_0 = KDTree(obj_vertex[obj_part_idx].cpu())
        nn = kdt_0.query(ref_frame.cpu().unsqueeze(0), k=1)[1][0,0]
        obj_sub_idx = torch.where(obj_adj_label == obj_adj_label[obj_part_idx[nn]])[0]
        kdt = KDTree(obj_vertex[obj_sub_idx].cpu())
        return obj_vertex[obj_sub_idx], obj_normal[obj_sub_idx], kdt

    for part_i in range(part_cnt):
        obj_vertex = torch.from_numpy(objs[part_i].vertices.copy()).cuda()
        obj_normal = torch.from_numpy(objs[part_i].vertex_normals.copy()).cuda()
        kdt_all = KDTree(obj_vertex.cpu())
        for (idx, hand_part_id) in enumerate([1,2,3,4,5,15,3,6,12,9]):
            finger_id = idx if idx < 5 else idx - 5
            for stage_i in range(c_flag.shape[0]):
                # Determine type
                aval_1 = stage_i != 0 and ref_flag[stage_i - 1, part_i, finger_id] > 0.5
                aval_2 = ref_flag[stage_i, part_i, finger_id] > 0.5
                if aval_1:
                    obj_part_verts_l, obj_part_norms_l, kdt_l = get_partial_mesh(objs[part_i], ref[stage_i - 1, part_i, finger_id, :3], ref[stage_i - 1, part_i, finger_id, 3:])
                if aval_2:
                    obj_part_verts_r, obj_part_norms_r, kdt_r = get_partial_mesh(objs[part_i], ref[stage_i, part_i, finger_id, :3], ref[stage_i, part_i, finger_id, 3:])
                verts_part_cur = verts_part if idx < 5 else verts_part_d
                for frame_i in range(stage_length[stage_i]):
                    # Compute optimize target
                    frame_id = torch.sum(stage_length[:stage_i]) + frame_i
                    finger_part_idx = torch.where(verts_part_cur == hand_part_id)[0]
                    finger_part_verts = hand_verts[frame_id][finger_part_idx]
                    finger_part_pf = finger_part_verts.clone().detach().cpu()
                    finger_part_pf = R.from_rotvec(obj_traj[stage_i, part_i, frame_i, 3:].cpu()).inv().apply(finger_part_pf - obj_traj[stage_i, part_i, frame_i, :3].cpu())
                    if (not aval_1 and not aval_2) or c_flag[stage_i, part_i, frame_i, finger_id] < 0.8:
                        obj_part_verts, obj_part_norms, kdt, c_type = obj_vertex, obj_normal, kdt_all, False
                        ref_norm = None
                    elif not aval_2 or (aval_1 and frame_i < stage_length[stage_i] / 2):
                        obj_part_verts, obj_part_norms, kdt, c_type = obj_part_verts_l, obj_part_norms_l, kdt_l, True
                        ref_norm = ref[stage_i - 1, part_i, finger_id, 3:]
                    else:
                        obj_part_verts, obj_part_norms, kdt, c_type = obj_part_verts_r, obj_part_norms_r, kdt_r, True
                        ref_norm = ref[stage_i, part_i, finger_id, 3:]
                    nn = torch.from_numpy(kdt.query(finger_part_pf, k=1)[1]).cuda()
                    target_verts = obj_part_verts[nn.squeeze()]
                    target_norms = obj_part_norms[nn.squeeze()]
                    finger_part_pf = torch.from_numpy(finger_part_pf).cuda()
                    displace = finger_part_pf - target_verts
                    if not c_type: # Non-contact: use whole mesh
                        inside = torch.from_numpy(obj_sdf[part_i].contains(finger_part_pf.detach().cpu().numpy())).cuda()
                        insiders = torch.where(inside == True)[0]
                        if (insiders.shape[0] < 1):
                            continue
                        dist = torch.sqrt(torch.sum(torch.square(displace), axis=1))
                        dist[insiders] *= -1
                    else: # Contact: use partial mesh + obj normal
                        if idx < 5:
                            continue
                        dist = torch.sum(displace * ref_norm, axis=1)
                    dist = (dist - torch.min(dist)) / 0.002
                    part_coef = torch.exp(-torch.square(dist))
                    opt_hand_idx.append(frame_id*778 + finger_part_idx)
                    target_verts_pf = torch.from_numpy(R.from_rotvec(obj_traj[stage_i, part_i, frame_i, 3:].cpu()).apply(target_verts.cpu())).cuda() + obj_traj[stage_i, part_i, frame_i, :3]
                    opt_target.append(target_verts_pf)
                    opt_coef.append(part_coef)

    opt_hand_idx = torch.cat(opt_hand_idx, dim=0)
    opt_target = torch.cat(opt_target, dim=0)
    opt_coef = torch.cat(opt_coef, dim=0)


    # Optimization
    new_pose_t = pose[1:].clone().detach().requires_grad_(True)
    new_trans_t = trans[1:].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([new_trans_t, new_pose_t], lr=0.01)

    for opt_step in range(500):
        opt.zero_grad()
        new_pose = torch.cat([pose[:1], new_pose_t], dim=0)
        new_trans = torch.cat([trans[:1], new_trans_t], dim=0)

        mano_output = mano_layer(new_pose[:, :3], new_pose[:, 3:])
        new_verts = mano_output.verts - mano_output.joints[:, 0:1, :] + new_trans.unsqueeze(1)
        new_joints = mano_output.joints - mano_output.joints[:, 0:1, :] + new_trans.unsqueeze(1)
        opt_verts = new_verts.reshape(-1, 3)[opt_hand_idx]

        contact_loss = torch.sum(opt_coef * torch.sum(torch.square(opt_verts-opt_target), dim=1))
        trans_loss = torch.sum(torch.square(new_trans - init_trans))

        joint_smooth1 = torch.sum(torch.square(new_joints[:-1] - new_joints[1:]))
        joint_smooth2 = torch.sum(torch.square(new_joints[:-2] - 2 * new_joints[1:-1] + new_joints[2:]))

        loss = contact_loss * 80 + trans_loss + (joint_smooth1 * 5 + joint_smooth2 * 20) * smooth_coef
        loss.backward()
        opt.step()

    new_pose = torch.cat([pose[:1], new_pose_t], dim=0)
    new_trans = torch.cat([trans[:1], new_trans_t], dim=0)

    return new_trans.detach(), new_pose.detach()
