import math

import numpy as np
from torch.autograd import Variable
from model.modules import *
from model.lib import ST_RenovateNet


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):
    def build_basic_blocks(self):
        A = self.graph.A  # 3,25,25
        self.l1 = TCN_GCN_unit(self.in_channels, self.base_channel, A, residual=False, adaptive=self.adaptive)
        self.l2 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l3 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l4 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l5 = TCN_GCN_unit(self.base_channel, self.base_channel * 2, A, stride=2, adaptive=self.adaptive)
        self.l6 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l7 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l8 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 4, A, stride=2, adaptive=self.adaptive)
        self.l9 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)
        self.l10 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(self.base_channel, self.num_frame, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_mid = ST_RenovateNet(self.base_channel * 2, self.num_frame // 2, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_high = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
            self.ren_fin = ST_RenovateNet(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.cl_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def __init__(self,
                 # Base Params
                 num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 base_channel=64, drop_out=0, adaptive=True,
                 # Module Params
                 cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', pred_threshold=0, use_p_map=True,
                 ):
        super(Model, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x
        self.adaptive = adaptive
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.build_basic_blocks()

        if self.cl_mode is not None:
            self.build_cl_blocks()

        self.fc = nn.Linear(self.base_channel * 4, self.num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def get_hidden_feat(self, x, pooling=True, raw=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # First stage
        x = self.l1(x)

        # Second stage
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        # Third stage
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        # Forth stage
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)

        if raw:
            return x

        if pooling:
            return x.mean(3).mean(1)
        else:
            return x.mean(1)

    def get_ST_Multi_Level_cl_output(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                  cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return logits, cl_loss

    def forward(self, x, label=None, get_cl_loss=False, get_hidden_feat=False, **kwargs):

        if get_hidden_feat:
            return self.get_hidden_feat(x)

        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        feat_low = x.clone()

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        feat_mid = x.clone()

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        feat_high = x.clone()

        x = self.l9(x)
        x = self.l10(x)
        feat_fin = x.clone()

        # N*M,C,T*V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return self.get_ST_Multi_Level_cl_output(x, feat_low, feat_mid, feat_high, feat_fin, label)

        return self.fc(x)
