import numpy as np
import torch
import torch.nn as nn
import torchgeometry as tgm

from .bone_pairs import get_vec_by_pose, get_pose_by_vec, get_sym_bone_matrix


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Linear(nn.Module):
    # linear -> bn -> relu -> linear -> bn -> relu
    def __init__(self, linear_size):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(inplace=True)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)

        return y


class PoseGenerator(nn.Module):
    def __init__(self, window_size=64, num_joints=25, squeeze_mode="average", aug_sub=True, aug_view=True,
                 aug_tmpl=False, use_T=False, rot_theta=[.3, .3, .3], T_theta=[.5, .5, .5], bone_theta=.2,
                 input_size=None, debug=False):
        super(PoseGenerator, self).__init__()
        self.window_size = window_size
        self.num_joints = num_joints
        self.squeeze_mode = squeeze_mode
        self.aug_sub = aug_sub
        self.aug_view = aug_view
        self.aug_tmpl = aug_tmpl
        self.debug = debug

        if input_size is None:
            self.input_size = num_joints * 3
            if self.squeeze_mode == "concat":
                self.input_size *= window_size
            self.input_size *= 2
        else:
            self.input_size = input_size

        if self.aug_sub:    # same bone len change for people in the same frame
            self.sub_gen = SubjectGenerator(input_size=self.input_size, bone_theta=bone_theta, debug=self.debug)
        if self.aug_view:   # same view change for people in the same frame
            self.view_gen = ViewGenerator(input_size=self.input_size, use_T=use_T, rot_theta=rot_theta, T_theta=T_theta,
                                          debug=self.debug)
        if self.aug_tmpl:
            # TODO: temporal augment
            self.tmpl_gen = None
            raise NotImplementedError("TODO temporal generator")

    def squeeze_time(self, data):
        """     squeeze the time dim
        Args:
            data (torch.Tensor): shape [N, C, T, V, M]
        Returns:
            torch.Tensor: shape [N, C', V, M]
        """
        N, C, T, V, M = data.shape
        if self.squeeze_mode == "random":
            select_frame = np.random.randint(T)
            return data[:, :, select_frame, :, :]
        elif self.squeeze_mode == "average":
            return data.mean(dim=2, keepdim=False)
        elif self.squeeze_mode == "concat":
            return data.view(N, C * T, V, M)
        else:
            raise NotImplementedError(f"No such squeeze_mode: {self.squeeze_mode}")

    def forward(self, inputs_data, inputs_feat=None):
        """
        Args:
            inputs_data: shape [N, C, T, V, M]
            inputs_feat: shape [N, C']
        Returns:
            torch.Tensor: augmented data, shape [N, C, T, V, M]
        """
        N, C, T, V, M = inputs_data.shape
        aug_data = inputs_data
        if inputs_feat is None:
            inputs_feat = self.squeeze_time(inputs_data).view(N, -1)

        ret_dict = dict()
        ret_dict['finer_aug_data'] = []

        if self.aug_sub:
            aug_data, blr = self.sub_gen(inputs_feat, aug_data)
            ret_dict['bone_len_args'] = blr
            ret_dict['finer_aug_data'].append(aug_data)
        if self.aug_view:
            aug_data, r, t = self.view_gen(inputs_feat, aug_data)
            ret_dict['rot_args'] = r
            if t is not None:
                ret_dict['mov_args'] = t
            ret_dict['finer_aug_data'].append(aug_data)
        if self.aug_tmpl:
            raise NotImplementedError("TODO temporal generator")

        ret_dict['aug_data'] = aug_data
        return ret_dict


# TODO: use LSTM for sub_aug and view_aug
class SubjectGenerator(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_stage=2, p_dropout=.5, bone_theta=.2, debug=False):
        super(SubjectGenerator, self).__init__()

        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_size = input_size
        self.bone_theta = bone_theta
        self.debug = debug

        # TODO: support for 'concat' squeeze_mode
        # 3d joints + bone_length
        self.input_size = input_size + 24  # 25 * 3 * 2 + 24

        # process input to linear size -> for R
        # TODO: noise_size use add operation
        self.w1_BL = nn.Linear(self.input_size + self.noise_size, self.hidden_size)
        self.batch_norm_BL = nn.BatchNorm1d(self.hidden_size)

        self.linear_stages_BL = []
        for l in range(num_stage):
            self.linear_stages_BL.append(Linear(self.hidden_size))
        self.linear_stages_BL = nn.ModuleList(self.linear_stages_BL)

        # post processing
        self.w2_BL = nn.Linear(self.hidden_size, 24)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, aug_data):
        """
        Args:
            x: feat of input data, shape [N, C']
            aug_data: raw data to augment, shape [N, C, T, V, M]
        Returns:
            torch.Tensor: augmented data, shape [N, C, T, V, M]
            torch.Tensor: bone len change args, shape [N, V - 1]
        """
        # convert to [N * M * T, V, 3]
        N, C, T, V, M = aug_data.shape
        aug_data = aug_data.permute(0, 4, 2, 3, 1).contiguous().view(N * M * T, V, C)
        # convert to root relative
        root = aug_data[:, :1, :]
        aug_data = aug_data - root

        # get bone_len, bone_dir
        bones_len, bones_dir = get_vec_by_pose(aug_data)
        # remove the time-dim
        x_bones_len = bones_len.squeeze(-1).view(N, M, T, V - 1).mean(dim=[1, 2], keepdim=False)   # [N, V - 1]

        # pre-processing inputs for network
        x_noise = torch.randn(x.shape[0], self.noise_size, device=x.device)

        # calculate scale ratio for bone length
        blr = self.w1_BL(torch.cat((x, x_bones_len, x_noise), dim=1))
        blr = self.batch_norm_BL(blr)
        blr = self.relu(blr)
        for i in range(self.num_stage):
            blr = self.linear_stages_BL[i](blr)
        blr = self.w2_BL(blr)
        # allow +-20% length change
        blr = nn.Tanh()(blr) * self.bone_theta + 1  # shape: [N, 24]
        tmp_blr = blr - 1
        # same for the T-dim
        blr = torch.cat([blr.unsqueeze(1) for _ in range(M)], dim=1)
        blr = torch.cat([blr.unsqueeze(2) for _ in range(T)], dim=2)
        blr = blr.view(N * M * T, -1)
        # shape: [N * M * T, 24]

        # calculate the augmented data
        scale = torch.matmul(get_sym_bone_matrix().to(blr.device), blr.unsqueeze(-1))      # make left and right bone equal
        bones_len = bones_len * scale
        aug_data = get_pose_by_vec(bones_dir * bones_len)
        aug_data = aug_data + root

        # convert back to [N, C, T, V, M]
        aug_data = aug_data.view(N, M, T, V, C).permute(0, 4, 2, 3, 1).contiguous()
        return aug_data, tmp_blr


class ViewGenerator(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_stage=2, p_dropout=.5,
                 use_T=False, rot_theta=[.3, .3, .3], T_theta=[.5, .5, .5], debug=False):
        # X' = rot_R * X + T, R:[3] -> rot_R:[3,3], T:[3]
        super(ViewGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.noise_size = input_size
        self.use_T = use_T
        self.rot_theta = rot_theta
        self.T_theta = T_theta
        self.debug = debug

        # 3d joints in one frame
        self.input_size = input_size  # 25 * 3 * 2

        # process input to linear size -> for R
        # TODO: noise_size use add operation
        self.w1_R = nn.Linear(self.input_size + self.noise_size, self.hidden_size)
        self.batch_norm_R = nn.BatchNorm1d(self.hidden_size)

        self.linear_stages_R = []
        for l in range(num_stage):
            self.linear_stages_R.append(Linear(self.hidden_size))
        self.linear_stages_R = nn.ModuleList(self.linear_stages_R)

        self.w2_R = nn.Linear(self.hidden_size, 3)      # final projection

        # process input to linear size -> for T
        if self.use_T:
            self.w1_T = nn.Linear(self.input_size + self.noise_size, self.hidden_size)
            self.batch_norm_T = nn.BatchNorm1d(self.hidden_size)

            self.linear_stages_T = []
            for l in range(num_stage):
                self.linear_stages_T.append(Linear(self.hidden_size))
            self.linear_stages_T = nn.ModuleList(self.linear_stages_T)

            self.w2_T = nn.Linear(self.hidden_size, 3)    # final projection

        self.relu = nn.LeakyReLU(inplace=True)
        # self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x, aug_data):
        """
        Args:
            x: feat of input data, shape [N, C']
            aug_data: raw data to augment, shape [N, C, T, V, M]
        Returns:
            torch.Tensor: augmented data, shape [N, C, T, V, M]
            torch.Tensor: rotate tensor, shape: [N, 3]
        """
        # convert to [N * M * T, V, 3]
        N, C, T, V, M = aug_data.shape
        aug_data = aug_data.permute(0, 4, 2, 3, 1).contiguous().view(N * M * T, V, C)

        # pre-processing inputs for network

        # calculate R
        noise = torch.randn(x.shape[0], self.noise_size, device=x.device)
        r = self.w1_R(torch.cat((x, noise), dim=1))
        r = self.batch_norm_R(r)
        r = self.relu(r)
        # r = self.dropout(r)
        for i in range(self.num_stage):
            r = self.linear_stages_R[i](r)

        r = self.w2_R(r)
        r = nn.Tanh()(r) * torch.Tensor(self.rot_theta).to(r.device)   # limit rotation range
        if self.debug:
            r = torch.Tensor(self.rot_theta).to(r.device).expand(N, 3)
        r = r.view(x.size(0), 3)            # r: [N, 3]
        tmp_r = r
        rM = tgm.angle_axis_to_rotation_matrix(r)[..., :3, :3]  # Nx4x4->Nx3x3 rotation matrix, rM: [N, 3, 3]
        rM = torch.cat([rM.unsqueeze(1) for _ in range(M)], dim=1)
        rM = torch.cat([rM.unsqueeze(2) for _ in range(T)], dim=2)
        rM = rM.view(N * M * T, 3, 3)
        # do same for M * T, rM: [N * M * T, 3, 3]
        aug_data = torch.matmul(rM, aug_data.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()

        # calculate T
        tmp_t = None
        if self.use_T:
            noise = torch.randn(x.shape[0], self.noise_size, device=x.device)
            t = self.w1_T(torch.cat((x, noise), dim=1))
            t = self.batch_norm_T(t)
            t = self.relu(t)
            for i in range(self.num_stage):
                t = self.linear_stages_T[i](t)

            t = self.w2_T(t)        # t: [N, 3]
            # t[:, 2] = t[:, 2].clone() * t[:, 2].clone()     # square on z-axis ? for positive move on z-axis ?
            t = nn.Tanh()(t) * torch.Tensor(self.T_theta).to(t.device)
            if self.debug:
                t = torch.Tensor(self.T_theta).to(t.device).expand(N, 3)
            tmp_t = t
            t = t.view(x.size(0), 1, 3)  # Nx1x3 translation t, t: [N, 1, 3]
            t = torch.cat([t.unsqueeze(1) for _ in range(M)], dim=1)
            t = torch.cat([t.unsqueeze(2) for _ in range(T)], dim=2)
            t = t.view(N * M * T, 1, 3)
            # do same for M * T, t: [N * M * T, 1, 3]
            aug_data = aug_data + t

        # convert back to [N, C, T, V, M]
        aug_data = aug_data.view(N, M, T, V, C).permute(0, 4, 2, 3, 1).contiguous()
        return aug_data, tmp_r, tmp_t


if __name__ == '__main__':
    pass
