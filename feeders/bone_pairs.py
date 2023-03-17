import torch

ntu_upper = (
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24  # 15 joints
)  # joint id start from 0, spine is 1

ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)  # joint id start from 1

ntu_dir_pairs = (
    (1, 2), (2, 21), (21, 3), (3, 4),
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 23), (23, 22),
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (25, 24),
    (1, 13), (13, 14), (14, 15), (15, 16),
    (1, 17), (17, 18), (18, 19), (19, 20),
)  # joint id start from 1, directional, root is 1, order is matter

ntu_sym_pairs = (
    (4, 10), (5, 11), (6, 12), (7, 13), (8, 14), (9, 15),
    (16, 20), (17, 21), (18, 22), (19, 23),
)  # bone id start from 0, symmetric pairs


def get_pose2vec_matrix(bone_pairs=ntu_dir_pairs, num_joints=25):
    r""" get transfer matrix for transfer 3D pose to 3D direction vectors.

    Returns:
        torch.Tensor: transfer matrix, shape like [num_joints - 1, num_joints]
    """
    matrix = torch.zeros((num_joints - 1, num_joints))  # [V - 1, V] * [V, 3] => [V - 1, 3]
    for i, (u, v) in enumerate(bone_pairs):
        matrix[i, u - 1] = -1
        matrix[i, v - 1] = 1
    return matrix


def get_vec2pose_matrix(bone_pairs=ntu_dir_pairs, num_joints=25):
    r""" get transfer matrix for transfer 3D direction vectors to 3D pose.

    Returns:
        torch.Tensor: transfer matrix, shape like [num_joints, num_joints - 1]
    """
    matrix = torch.zeros((num_joints, num_joints - 1))  # [V, V - 1] * [V - 1, 3] => [V, 3]
    for i, (u, v) in enumerate(bone_pairs):
        matrix[v - 1, :] = matrix[u - 1, :]
        matrix[v - 1, i] = 1
    return matrix


def get_sym_bone_matrix(sym_pairs=ntu_sym_pairs, num_joints=25):
    r""" get transfer matrix for average the left and right bones

    Returns:
        torch.Tensor: transfer matrix, shape like [num_joints - 1, num_joints - 1]
    """
    matrix = torch.zeros((num_joints - 1, num_joints - 1))  # [V - 1, V - 1] * [V - 1, 1] => [V - 1, 1]
    for i in range(num_joints - 1):
        matrix[i, i] = 1
    for (i, j) in sym_pairs:
        matrix[i, i] = matrix[i, j] = 0.5
        matrix[j, j] = matrix[j, i] = 0.5
    return matrix


def get_vec_by_pose(joints):
    r""" get unit bone vec & bone len from joints

    Args:
        joints (torch.Tensor): relative to the root, shape like [num_joints, 3]
    Returns:
        torch.Tensor: unit bone vec, shape like [num_joints - 1, 3]
        torch.Tensor: bone len, shape like [num_joints - 1, 1]
    """
    bones = torch.matmul(get_pose2vec_matrix().to(joints.device), joints)
    bones_len = torch.norm(bones, dim=-1, keepdim=True)
    bones_dir = bones / (bones_len + 1e-8)
    return bones_len, bones_dir


def get_pose_by_vec(bones):
    r""" get joints from bone vec (not unit)

    Returns:
        torch.Tensor: relative to the root, shape like [num_joints, 3]
    """
    return torch.matmul(get_vec2pose_matrix().to(bones.device), bones)


###### group skeleton representation
ntu_groups = (
    (0, 1, 20),     # spine
    (2, 3),         # head
    (8, 9, 10, 11, 23, 24),    # left arm
    (4, 5, 6, 7, 21, 22),       # right arm
    (16, 17, 18, 19),           # left leg
    (12, 13, 14, 15),           # right leg
)   # joint id start from 0
ntu_group_roots = (-1, 20, 20, 20, 0, 0)    # origin from each group, start from 0


def ske2group(x):
    # x: [N, C, T, V, M]
    ret = x.clone()
    for group, root in zip(ntu_groups, ntu_group_roots):
        if root == -1:
            continue
        ret[:, :, :, group, :] -= x[:, :, :, root:root+1, :]
    return ret


def group2ske(x):
    # x: [N, C, T, V, M]
    ret = x.clone()
    for group, root in zip(ntu_groups, ntu_group_roots):
        if root == -1:
            continue
        ret[:, :, :, group, :] += x[:, :, :, root:root+1, :]
    return ret


# Module Testing
if __name__ == '__main__':
    joints_raw = torch.randn((100, 25, 3))
    root = joints_raw[:, :1, :]

    joints = joints_raw - root     # relative to the root joint

    bones_len, bones_dir = get_vec_by_pose(joints)
    joints_after = get_pose_by_vec(bones_dir * bones_len)

    joints_after = joints_after + root

    EPS = 1e-6
    print((torch.abs(joints_raw - joints_after) < EPS).sum() == 100 * 25 * 3)

    bones_len, bones_dir = get_vec_by_pose(joints)
    scale = torch.zeros(25 - 1).uniform_(-.2, .2) + 1
    scale = scale.unsqueeze(0).unsqueeze(-1)
    scale = torch.matmul(get_sym_bone_matrix(), scale)
    for (u, v) in ntu_sym_pairs:
        if abs(scale[0, u, 0] - scale[0, v, 0]) > EPS:
            print("Testing Failed")
