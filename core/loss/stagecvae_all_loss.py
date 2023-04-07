from turtle import forward
import torch
import torch.nn as nn
import trimesh

class CVAEALLLoss(nn.Module):
    def __init__(self, lambda_dir=100, lambda_seq_joints=1) -> None:
        super().__init__()
        self.lambda_binary = 0.1
        self.lambda_pos = 500
        self.lambda_dir = lambda_dir
        # self.lambda_dir = 1
        self.lambda_KLD = 5

        self.lambda_seq_tip = 100
        self.lambda_seq_joints = lambda_seq_joints

    def forward(self, batch):

        gt_ref = batch['contact_ref']
        gt_seq_1 = batch['out_seq_1']
        gt_seq_2 = batch['out_seq_2']
        gt_c = batch['out_contact_flag_seq']
        gt_n = batch['out_near_flag_seq']
        pred_ref_flag = batch['pred_ref_flag']
        pred_ref = batch['pred_ref']
        pred_seq_1 = batch['pred_seq_1']
        pred_seq_2 = batch['pred_seq_2']
        pred_c = batch['pred_c_flag']
        pred_n = batch['pred_n_flag']
        mu = batch['mu']
        logvar = batch['logvar']

        gt_flag = gt_ref[:, :, :, :, 0]
        gt_ref = gt_ref[:, :, :, :, 1:]

        binary_loss = (-gt_flag*(pred_ref_flag+1e-6).log()-(1-gt_flag)*(1-pred_ref_flag+1e-6).log()).mean() \
                    + (-gt_c*(pred_c+1e-6).log()-(1-gt_c)*(1-pred_c+1e-6).log()).mean() \
                    + (-gt_n*(pred_n+1e-6).log()-(1-gt_n)*(1-pred_n+1e-6).log()).mean()

        gt_flag = gt_flag.unsqueeze(-1)
        pos_L2_loss = (gt_flag*(gt_ref[:, :, :, :, :3]-pred_ref[:, :, :, :, :3])**2).sum()/gt_flag.sum()
        dir_L2_loss = (gt_flag*(gt_ref[:, :, :, :, 3:]-pred_ref[:, :, :, :, 3:])**2).sum()/gt_flag.sum()
        KLD_loss = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()

        gt_flag_1 = gt_flag[:, :-1].unsqueeze(-3)
        gt_flag_1 = torch.cat([torch.zeros_like(gt_flag_1[:, :1]), gt_flag_1], dim=1)
        gt_flag_2 = gt_flag.unsqueeze(-3)
        n_mask = gt_n.unsqueeze(-1)
        delta_seq_1 = (gt_flag_1 * n_mask * (gt_seq_1-pred_seq_1)**2).view(-1, 5, 3)
        delta_seq_2 = (gt_flag_2 * n_mask * (gt_seq_2-pred_seq_2)**2).view(-1, 5, 3)
        tip_loss = delta_seq_1[:, 4].mean() + delta_seq_2[:, 4].mean()
        joints_loss = delta_seq_1[:, :4].mean() + delta_seq_2[:, :4].mean()

        binary_loss = self.lambda_binary * binary_loss
        pos_L2_loss = self.lambda_pos * pos_L2_loss
        dir_L2_loss = self.lambda_dir * dir_L2_loss
        KLD_loss = self.lambda_KLD * KLD_loss
        tip_loss = self.lambda_seq_tip * tip_loss
        joints_loss = self.lambda_seq_joints * joints_loss

        return binary_loss, pos_L2_loss, dir_L2_loss, KLD_loss, tip_loss, joints_loss
