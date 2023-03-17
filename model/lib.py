import math
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F


class RenovateNet(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, h_channel=None, version='V0',
                 pred_threshold=0.0, use_p_map=True):
        super(RenovateNet, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class

        self.alp = alp
        self.tmp = tmp
        self.mom = mom

        self.avg_f = torch.randn(self.h_channel, self.n_class)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.version = version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map

    def onehot(self, label):
        # input: label: [N]; output: [N, K]
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class)
        return ones.view(*size).float()

    def get_mask_fn_fp(self, lbl_one, pred_one, logit):
        # input: [N, K]; output: tp,fn,fp:[N, K] has_fn,has_fp:[K, 1]
        tp = lbl_one * pred_one
        fn = lbl_one - tp
        fp = pred_one - tp

        tp = tp * (logit > self.pred_threshold).float()

        num_fn = fn.sum(0).unsqueeze(1)     # [K, 1]
        has_fn = (num_fn > 1e-8).float()
        num_fp = fp.sum(0).unsqueeze(1)     # [K, 1]
        has_fp = (num_fp > 1e-8).float()
        return tp, fn, fp, has_fn, has_fp

    def local_avg_tp_fn_fp(self, f, mask, fn, fp):
        # input: f:[N, C], mask,fn,fp:[N, K]
        b, k = mask.size()
        f = f.permute(1, 0)  # [C, N]
        avg_f = self.avg_f.detach().to(f.device)  # [C, K]

        fn = F.normalize(fn, p=1, dim=0)
        f_fn = torch.matmul(f, fn)  # [C, K]

        fp = F.normalize(fp, p=1, dim=0)
        f_fp = torch.matmul(f, fp)

        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask)
        # dist.all_reduce(f_mask, op=dist.reduce_op.SUM)
        # dist.all_reduce(mask_sum, op=dist.reduce_op.SUM)
        f_mask = f_mask / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()

        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem
        return f_mem, f_fn, f_fp

    def get_score(self, feature, lbl_one, logit, f_mem, f_fn, f_fp, s_fn, s_fp, mask_tp):
        # feat: [N, C], lbl_one,logit: [N, K], f_fn,f_fp,f_mem: [C, K], s_fn,s_fp:[K, 1], mask_tp: [N, K]
        # output: [K, N]

        (b, c), k = feature.size(), self.n_class

        feature = feature / (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)

        f_mem = f_mem.permute(1, 0)  # k,c
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)

        f_fn = f_fn.permute(1, 0)  # k,c
        f_fn = f_fn / (torch.norm(f_fn, p=2, dim=-1, keepdim=True) + 1e-12)
        f_fp = f_fp.permute(1, 0)  # k,c
        f_fp = f_fp / (torch.norm(f_fp, p=2, dim=-1, keepdim=True) + 1e-12)

        if self.use_p_map:
            p_map = (1 - logit) * lbl_one * self.alp  # N, K
        else:
            p_map = lbl_one * self.alp  # N, K

        score_mem = torch.matmul(f_mem, feature.permute(1, 0))  # K, N

        if self.version == "V0":
            score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1    # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map = score_fn * p_map.permute(1, 0) * s_fn
            fp_map = score_fp * p_map.permute(1, 0) * s_fp     # K, N

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        elif self.version == "V1":  # 只有TP 才有惩罚项
            score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map = score_fn * p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            fp_map = score_fp * p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        elif self.version == "NO FN":
            # score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            # fn_map = score_fn * p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            fp_map = score_fp * p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = score_mem / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        elif self.version == "NO FP":
            score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            # score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map = score_fn * p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            # fp_map = score_fp * p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = score_mem / self.tmp
        elif self.version == "NO FN & FP":
            # score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            # score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            # fn_map = score_fn * p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            # fp_map = score_fp * p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = score_mem / self.tmp
            score_cl_fp = score_mem / self.tmp
        elif self.version == "V2":  # 惩罚项计算的是 与均值直接的距离
            score_fn = torch.sum(f_mem * f_fn, dim=1, keepdim=True) - 1    # K, 1
            score_fp = - torch.sum(f_mem * f_fp, dim=1, keepdim=True) - 1  # K, 1
            fn_map = score_fn * s_fn
            fp_map = score_fp * s_fp  # K, 1

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        else:
            score_cl_fn, score_cl_fp = None, None


        return score_cl_fn, score_cl_fp

    def forward(self, feature, lbl, logit, return_loss=True):
        # feat: [N, C], lbl: [N], logit: [N, K]
        # output: [N, K]
        feature = self.cl_fc(feature)
        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)

        logit = torch.softmax(logit, 1)
        mask, fn, fp, has_fn, has_fp = self.get_mask_fn_fp(lbl_one, pred_one, logit)
        f_mem, f_fn, f_fp = self.local_avg_tp_fn_fp(feature, mask, fn, fp)
        score_cl_fn, score_cl_fp = self.get_score(feature, lbl_one, logit, f_mem, f_fn, f_fp, has_fn, has_fp, mask)

        score_cl_fn = score_cl_fn.permute(1, 0).contiguous()    # [N, K]
        score_cl_fp = score_cl_fp.permute(1, 0).contiguous()    # [N, K]
        p_map = ((1 - logit) * lbl_one).sum(dim=1)  # N

        if return_loss:
            if self.version in ["V0", "V1", "NO FN", "NO FP", "NO FN & FP"]:
                return (self.loss(score_cl_fn, lbl) + self.loss(score_cl_fp, lbl)).mean()
            else:
                return (p_map * self.loss(score_cl_fn, lbl) + p_map * self.loss(score_cl_fp, lbl)).mean()
        else:
            return score_cl_fn.permute(1, 0).contiguous(), score_cl_fp.permute(1, 0).contiguous()


class ST_RenovateNet(nn.Module):
    def __init__(self, n_channel, n_frame, n_joint, n_person, h_channel=256, **kwargs):
        super(ST_RenovateNet, self).__init__()
        self.n_channel = n_channel
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.n_person = n_person

        self.spatio_cl_net = RenovateNet(n_channel=h_channel // n_joint * n_joint, h_channel=h_channel, **kwargs)
        self.tempor_cl_net = RenovateNet(n_channel=h_channel // n_frame * n_frame, h_channel=h_channel, **kwargs)

        self.spatio_squeeze = nn.Sequential(nn.Conv2d(n_channel, h_channel // n_joint, kernel_size=1),
                                            nn.BatchNorm2d(h_channel // n_joint), nn.ReLU(True))
        self.tempor_squeeze = nn.Sequential(nn.Conv2d(n_channel, h_channel // n_frame, kernel_size=1),
                                            nn.BatchNorm2d(h_channel // n_frame), nn.ReLU(True))


    def forward(self, raw_feat, lbl, logit, **kwargs):
        # raw_feat: [N * M, C, T, V]
        raw_feat = raw_feat.view(-1, self.n_person, self.n_channel, self.n_frame, self.n_joint)

        spatio_feat = raw_feat.mean(1).mean(-2, keepdim=True)
        spatio_feat = self.spatio_squeeze(spatio_feat)
        spatio_feat = spatio_feat.flatten(1)
        spatio_cl_loss = self.spatio_cl_net(spatio_feat, lbl, logit, **kwargs)

        tempor_feat = raw_feat.mean(1).mean(-1, keepdim=True)
        tempor_feat = self.tempor_squeeze(tempor_feat)
        tempor_feat = tempor_feat.flatten(1)
        tempor_cl_loss = self.tempor_cl_net(tempor_feat, lbl, logit, **kwargs)

        return spatio_cl_loss + tempor_cl_loss
