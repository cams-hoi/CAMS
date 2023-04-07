import math

import torch
import torch.nn as nn

__all__ = ['CAMS_CVAE']

class TemporalEncoding(nn.Module):
    def __init__(self, L) -> None:
        super().__init__()
        self.coeff = torch.tensor([2**i*math.pi for i in range(L)]).view(1, -1).cuda()

    def forward(self, t):

        # batch_size, seq_len = t.shape
        t = t.view(-1, 1)-0.5
        emb_sin = torch.sin(t*self.coeff)
        emb_cos = torch.cos(t*self.coeff)
        y = torch.cat([emb_sin, emb_cos], dim=1)
        # y = y.view(batch_size, seq_len, -1)
        return y

class PointNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_channels, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.activate = nn.ELU()

    # x: N * C * L
    def forward(self, x):
        x = self.activate(self.bn1(self.conv1(x)))
        x = self.activate(self.bn2(self.conv2(x)))
        x = self.activate(self.bn3(self.conv3(x)))

        return torch.max(x, dim=2)[0]

class CAMS_CVAE(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()

        shape_feature_dim=model_config.shape_feature_dim
        latent_dim=model_config.latent_dim
        temporal_dim=model_config.temporal_dim
        in_seq_dim=model_config.in_seq_dim
        out_seq_dim=model_config.out_seq_dim
        control_dim=model_config.control_dim
        condition_dim=model_config.condition_dim
        seq_len=model_config.seq_len
        ref_dim=model_config.ref_dim
        n_stages=model_config.n_stages
        n_parts=model_config.n_parts
        input_obj_dim=model_config.input_obj_dim
        self.n_stages = model_config.n_stages
        self.n_parts = model_config.n_parts
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.out_seq_dim = out_seq_dim

        self.activation = nn.ELU()
        self.pointnet_enc = PointNetEncoder(input_obj_dim, shape_feature_dim)
        self.condition_enc = nn.Sequential(
            nn.Linear(self.n_parts*shape_feature_dim+control_dim, condition_dim),
            nn.BatchNorm1d(condition_dim),
            nn.ELU()
        )

        self.mlp_vae_enc = nn.Sequential(
            nn.Linear(condition_dim+seq_len*ref_dim*n_stages*n_parts+seq_len*in_seq_dim*n_stages*n_parts, 1024), nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ELU()
        )

        self.enc_fc_mu = nn.Linear(1024, latent_dim)
        self.enc_fc_var = nn.Linear(1024, latent_dim)

        self.state_dec = nn.ModuleList([nn.Sequential(
            nn.Linear(condition_dim+latent_dim, 1024),
            nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ELU(),
            nn.Linear(2048, ref_dim)
        ) for _ in range(n_stages*n_parts)])

        self.seq_dec = nn.ModuleList([nn.Sequential(
            nn.Linear(condition_dim+latent_dim+temporal_dim, 1024),
            nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ELU(),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ELU(),
            nn.Linear(2048, out_seq_dim)
        ) for _ in range(n_stages*n_parts)])

        self.temp_enc = TemporalEncoding(3)
        # self.dropout_shape_feature = nn.Dropout(0.5)

    def forward(self, batch):

        obj_vertices = batch['obj_vertices']
        control = batch['control']

        condition = self.encode_condition(obj_vertices, control)

        ref = batch['contact_ref']
        in_seq_1 = batch['in_seq_1']
        in_seq_2 = batch['in_seq_2']
        in_contact_flag = batch['in_contact_flag_seq']
        in_near_flag = batch['in_near_flag_seq']

        mu, logvar = self.encode(condition, ref, in_seq_1, in_seq_2, in_contact_flag, in_near_flag)

        batch['mu'] = mu
        batch['logvar'] = logvar

        latent = self.reparam(mu, logvar)

        # latent = latent.view(-1, 1, latent_dim).tile((1, out_L, 1)).view(-1, latent_dim)
        timestamp = batch['t_seq']
        time_emb = self.temp_enc(timestamp)

        pred_ref_flag, pred_ref, pred_seq_1, pred_seq_2, \
        pred_c_flag, pred_n_flag = self.decode(condition, latent, time_emb)

        batch['pred_ref_flag'] = pred_ref_flag
        batch['pred_ref'] = pred_ref
        batch['pred_seq_1'] = pred_seq_1
        batch['pred_seq_2'] = pred_seq_2
        batch['pred_c_flag'] = pred_c_flag
        batch['pred_n_flag'] = pred_n_flag

        return batch

    def inference(self, batch):

        obj_vertices = batch['obj_vertices']
        control = batch['control']

        batch_size = obj_vertices.shape[0]

        condition = self.encode_condition(obj_vertices, control)

        latent = torch.randn((batch_size, self.latent_dim)).cuda()
        timestamp = torch.linspace(0, 1, 100).cuda().view(1, 1, 1, 100, -1).tile((batch_size, self.n_stages, self.n_parts, 1, 1))
        time_emb = self.temp_enc(timestamp)

        pred_ref_flag, pred_ref, pred_seq_1, pred_seq_2, \
        pred_c_flag, pred_n_flag = self.decode(condition, latent, time_emb)

        batch['pred_ref_flag'] = pred_ref_flag
        batch['pred_ref'] = pred_ref
        batch['pred_seq_1'] = pred_seq_1
        batch['pred_seq_2'] = pred_seq_2
        batch['pred_c_flag'] = pred_c_flag
        batch['pred_n_flag'] = pred_n_flag

        return batch

    def encode_condition(self, in_shape, in_cond):
        shape_feature = []
        for part in range(self.n_parts):
            in_part = in_shape[:, part].permute(0, 2, 1)
            shape_feature.append(self.pointnet_enc(in_part))
        shape_feature.append(in_cond)
        condition = self.condition_enc(torch.cat(shape_feature, dim=1))
        return condition

    def encode(self, condition, contact_ref, in_seq_1, in_seq_2, in_contact_flag, in_near_flag):
        B = condition.shape[0]

        in_seq = torch.cat([
            in_seq_1, in_seq_2, in_contact_flag.unsqueeze(-1), in_near_flag.unsqueeze(-1)
        ], dim=-1)

        contact_ref = contact_ref.unsqueeze(3).tile(1, 1, 1, self.seq_len, 1, 1)
        seq = torch.cat([contact_ref, in_seq], dim=-1)
        # f = self.mlp_vae_enc(torch.cat([
        #     condition, contact_ref.view(B, -1), in_seq.view(B, -1)
        # ], dim=1))
        f = self.mlp_vae_enc(torch.cat([
            condition, seq.view(B, -1)
        ], dim=1))

        return self.enc_fc_mu(f), self.enc_fc_var(f)

    def reparam(self, mu, logvar):
        std_var = torch.exp(0.5*logvar)
        return mu + std_var * torch.randn_like(std_var)

    def decode(self, condition, latent, time_emb):
        dec_in = torch.cat([condition, latent], dim=1)
        time_emb = time_emb.view(latent.shape[0], self.n_stages, self.n_parts, -1, time_emb.shape[-1])
        # time_emb: B * S * P * L * T

        state_out = torch.cat([f(dec_in) for f in self.state_dec], dim=1).view(
            -1, self.n_stages, self.n_parts, 5, 7
        )

        seq_out = []
        for stage in range(self.n_stages):
            stage_out = []
            for part in range(self.n_parts):
                f = self.seq_dec[stage*self.n_parts+part]

                tmp_in = dec_in.unsqueeze(1).tile((1, time_emb.shape[-2], 1))
                tmp_in = torch.cat([tmp_in, time_emb[:, stage, part]], dim=-1)

                stage_out.append(f(tmp_in.view(-1, tmp_in.shape[-1])).view(-1, time_emb.shape[-2], 5, self.out_seq_dim//5))
            seq_out.append(torch.stack(stage_out, dim=1))

        seq_out = torch.stack(seq_out, dim=1)
        out_seq_1 = seq_out[:, :, :, :, :, :15]
        out_seq_2 = seq_out[:, :, :, :, :, 15:30]
        out_contact_flag = torch.sigmoid(seq_out[:, :, :, :, :, 30])
        out_near_flag = torch.sigmoid(seq_out[:, :, :, :, :, 31])

        return (torch.sigmoid(state_out[:, :, :, :, 0]), state_out[:, :, :, :, 1:],
                out_seq_1, out_seq_2, out_contact_flag, out_near_flag)
