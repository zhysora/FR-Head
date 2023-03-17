import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rectified_l2_loss(gamma, threshold):  # threshold = b
    diff = (gamma - 0) ** 2
    weight = torch.where(diff > threshold ** 2, torch.ones_like(gamma), torch.zeros_like(gamma))
    diff_weighted = diff * weight
    return diff_weighted.mean()


def get_reg_loss(arg, epoch, aug_dict):
    """  control the modification range  """
    reg_loss = 0
    if 'bone_len_args' in aug_dict:
        blr = aug_dict['bone_len_args']
        blr_loss = rectified_l2_loss(blr, arg.bone_len_diff_limit)
        reg_loss += blr_loss.mean()
    if 'rot_args' in aug_dict:
        rot = aug_dict['rot_args']
        rot_loss = rectified_l2_loss(rot, arg.rot_diff_limit)
        reg_loss += rot_loss.mean()
    if 'mov_args' in aug_dict:
        mov = aug_dict['mov_args']
        mov_loss = rectified_l2_loss(mov, arg.mov_diff_limit)
        reg_loss += mov_loss.mean()
    return reg_loss


def diff_range_loss(a, b, std):
    diff = (a - b) ** 2
    weight = torch.where(diff > std ** 2, torch.ones_like(a), torch.zeros_like(a))
    diff_weighted = diff * weight
    return diff_weighted.mean()


def get_feed_back_loss(arg, epoch, real_loss, fake_loss, part_id, eps=1e-8):
    hard_ratio = arg.hard_ratio_st[part_id] + (arg.hard_ratio_ed[part_id] - arg.hard_ratio_st[part_id]) \
                 * (epoch - max(arg.with_aug_epoch, 0)) / (arg.num_epoch - max(arg.with_aug_epoch, 0))

    if not arg.control_std:  # exp loss
        return torch.abs(1 - torch.exp(fake_loss - hard_ratio * real_loss)).mean()
    else:  # control std & mean
        hard_value = fake_loss / (real_loss + eps)
        hard_std = torch.std(hard_value)

        hard_std_loss = torch.mean((hard_std - arg.target_std[part_id]) ** 2)
        hard_mean_loss = diff_range_loss(hard_value, hard_ratio, arg.target_std[part_id])

        return hard_std_loss * arg.w_fd_loss_std[part_id] + hard_mean_loss * arg.w_fd_loss_mean[part_id]


# DND
def calc_reweight_factor(f_x, f_aug_x, y, alpha=.5, beta=.5):
    N = f_x.shape[0]

    soft_x = F.softmax(f_x, dim=-1).gather(-1, y.view(N, 1))
    soft_aug_x = F.softmax(f_aug_x, dim=-1).gather(-1, y.view(N, 1))
    diff = soft_x - soft_aug_x
    return torch.pow(soft_aug_x, alpha) * torch.pow(torch.where(diff > 0, diff, torch.zeros_like(diff)), beta)


def calc_sim_loss(sem_x, sem_aug_x, classifier):
    sem_x, sem_aug_x = sem_x.detach(), sem_aug_x.detach()
    N = sem_x.shape[0]
    label_pos = torch.ones(N).long().to(sem_x.device)
    label_neg = torch.zeros(N * N - N).long().to(sem_x.device)

    pos_pair = torch.cat([sem_x, sem_aug_x, torch.abs(sem_x-sem_aug_x)], dim=-1)
    neg_pair = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            neg_pair.append(torch.cat([sem_x[i:i+1,:], sem_x[j:j+1,:], torch.abs(sem_x[i:i+1,:]-sem_x[j:j+1,:])], dim=-1))
    neg_pair = torch.cat(neg_pair, dim=0)

    loss_pos = F.cross_entropy(classifier(pos_pair), label_pos)
    loss_neg = F.cross_entropy(classifier(neg_pair), label_neg)
    return torch.mean(loss_pos + loss_neg)


def calc_reward_task(x, aug_x, y, model, loss_func, alpha=.5, beta=.5):
    # reward difficult samples
    N = aug_x.shape[0]

    f_x, f_aug_x = model(x), model(aug_x)
    soft_x = F.softmax(f_x, dim=-1).gather(-1, y.view(N, 1))
    soft_aug_x = F.softmax(f_aug_x, dim=-1).gather(-1, y.view(N, 1))

    loss_task = loss_func(f_aug_x, y)
    weight = torch.pow(soft_aug_x, alpha) * torch.pow(torch.max(soft_x - soft_aug_x, 0), beta)
    return torch.mean(weight * loss_task)


def calc_reward_sim(sem_x, sem_aug_x, classifier):
    # reward not too different samples
    N = sem_x.shape[0]
    label_pos = torch.ones(N).long().to(sem_x.device)
    return -F.cross_entropy(classifier(torch.cat([sem_x, sem_aug_x, torch.abs(sem_x-sem_aug_x)], dim=-1)), label_pos)


if __name__ == '__main__':
    pass
