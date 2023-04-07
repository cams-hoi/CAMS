from logging import root
from manotorch import manolayer
from manotorch import axislayer
import torch

class ManoLayer(torch.nn.Module):
    def __init__(self,
        right_side = True,
        mano_shape = [-0.2958,  0.9205, -0.3387, -2.2807, -0.8466, -1.1710, -2.0076,  1.6172, -0.9288,  0.4104],
        mano_assets_root='data/mano_assets/mano/',
        cuda = False
        ):

        super().__init__()
        self.mano_layer = manolayer.ManoLayer(
            mano_assets_root=mano_assets_root,
            side='right' if right_side else 'left',
            use_pca=False,
        )

        self.shape_param = torch.tensor(mano_shape).view((1, 10))
        zero_result = self.mano_layer(torch.zeros(1, 48), self.shape_param)
        axis_layer = axislayer.AxisLayer()
        t_axis, s_axis, b_axis = axis_layer(zero_result.joints, zero_result.transforms_abs)
        t_axis = torch.cat((torch.tensor([[1, 0, 0]]), t_axis.detach()[0])).view(-1, 1, 3)
        s_axis = torch.cat((torch.tensor([[0, 1, 0]]), s_axis.detach()[0])).view(-1, 1, 3)
        b_axis = torch.cat((torch.tensor([[0, 0, 1]]), b_axis.detach()[0])).view(-1, 1, 3)
        self.axis = torch.cat((t_axis, s_axis, b_axis), 1)

        self.th_faces = self.mano_layer.th_faces

        self.pose_lb = torch.tensor([
            0, -0.1, -0.0, 0, 0, 0, 0, 0, 0,
            0, -0.1, -0.0, 0, 0, 0, 0, 0, 0,
            0, -0.1, -0.0, 0, 0, 0, 0, 0, 0,
            0, -0.1, -0.0, 0, 0, 0, 0, 0, 0,
            0, -0.25, -0.15, 0, 0, 0, 0, 0, 0
        ], dtype=torch.float32) * torch.pi

        self.pose_ub = torch.tensor([
            0, 0.1, 0.5, 0, 0, 0.5, 0, 0, 0.5,
            0, 0.1, 0.5, 0, 0, 0.5, 0, 0, 0.5,
            0, 0.1, 0.5, 0, 0, 0.5, 0, 0, 0.5,
            0, 0.1, 0.5, 0, 0, 0.5, 0, 0, 0.5,
            0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0.5
        ], dtype=torch.float32) * torch.pi

        if cuda:
            self.pose_lb = self.pose_lb.cuda()
            self.pose_ub = self.pose_ub.cuda()
            self.axis = self.axis.cuda()
            self.mano_layer = self.mano_layer.cuda()
            self.shape_param = self.shape_param.cuda()

    def to_original_mano(self, rot, pose):
        real_pose = torch.sigmoid(pose) * (self.pose_ub-self.pose_lb) + self.pose_lb
        pose_coeffs = torch.cat((rot, real_pose), 1).view(-1, 16, 1, 3)

        new_axis = torch.matmul(pose_coeffs, self.axis).view(-1, 48)
        return new_axis

    def forward(self, rot:torch.Tensor, pose: torch.Tensor):
        new_axis = self.to_original_mano(rot, pose)
        shapes = torch.tile(self.shape_param, (new_axis.shape[0], 1))

        return self.mano_layer(new_axis, shapes)
