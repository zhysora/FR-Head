import numpy as np
import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from feeders.bone_pairs import ntu_pairs
from feeders.feeder_ntu import ntu120_class_name, ntu120_class_name_short


def draw_pose_3d(pose_3d, ax, colors=["#3498db", "#e74c3c"]):
    # pose_3d: [V, 3, 2], visualize 2 people
    for pair in ntu_pairs:
        for pid in [0, 1]:
            x, z, y = [np.array([pose_3d[pair[0]-1, j, pid], pose_3d[pair[1]-1, j, pid]]) for j in range(3)]
            ax.plot(x, y, z, lw=2, c=colors[pid])


def record_skeleton(data, label, writer, tag):
    # choose one item to visualize
    # data: [N, C, T, V, M]
    item = data[0].permute(1, 2, 0, 3).cpu().numpy()
    # item: [T, V, C, M]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    for i in range(0, item.shape[0], 8):    # skip by 8
        ax.lines = []
        draw_pose_3d(item[i], ax)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        img_array = np.asarray(Image.open(buffer))  # new_img就是figure的数组
        buffer.close()
        writer.add_image(tag + f'_{ntu120_class_name[label[0].cpu().numpy()]}', img_array, i, dataformats='HWC')

    plt.close()


def export_skeleton_frames(data, dir_path, prefix=""):
    # choose one item to visualize
    # data: [C, T, V, M]
    item = data.permute(1, 2, 0, 3).cpu().numpy()
    # item: [T, V, C, M]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    for i in range(0, item.shape[0]):
        ax.lines = []
        draw_pose_3d(item[i], ax)

        plt.savefig(f'{dir_path}/{prefix}{i}.png', format='png')

    plt.close()


def export_pred_bar(pred, file_path):
    # pred: [120]
    K = 5
    fontsize = 15
    bin_width = 0.5

    pred = torch.softmax(pred, dim=0)
    values, indices = torch.topk(pred, K)
    values = values.cpu().numpy()
    indices = indices.cpu().numpy()

    plt.figure(figsize=(3.2, 4.8))

    # print(values)
    # print([ntu120_class_name[int(indices[_])].split('.')[-1].strip() for _ in range(K)])
    # exit(0)
    def easy_name(str):
        str = str.split('.')[-1].strip()
        if '(' in str:
            str = str.split('(')[0] + str.split(')')[-1]

        ret = ""
        for id, c in enumerate(str):
            ret += c
            if (id + 1) % 15 == 0:
                ret += '\n'
        return ret

    plt.bar([_ for _ in range(K)], values * 100, width=bin_width, color=['#61d8e4'])
    plt.xticks([_ for _ in range(K)], [easy_name(ntu120_class_name_short[int(indices[_])]) for _ in range(K)],
               fontsize=fontsize, rotation=90, name="Times New Roman")

    plt.yticks(fontsize=fontsize, name="Times New Roman")
    plt.tight_layout()

    plt.savefig(f'{file_path}.png')
    plt.close()


def wrong_analyze(wf, rf):
    result_file = os.path.dirname(wf) + '/wrong_analyze'

    y_true = []
    y_pred = []
    with open(wf, 'r') as f:
        for line in f.readlines():
            data = line.split(',')
            y_true.append(int(data[-1]))
            y_pred.append(int(data[-2]))
    with open(rf, 'r') as f:
        for line in f.readlines():
            data = line.split(',')
            y_true.append(int(data[-1]))
            y_pred.append(int(data[-2]))

    class_num = max(y_true + y_pred) + 1

    CNT = np.zeros(class_num)
    TP = np.zeros(class_num)
    FN = np.zeros([class_num, class_num])

    for y0, y1 in zip(y_true, y_pred):
        CNT[y0] += 1
        if y0 == y1:
            TP[y0] += 1
        else:
            FN[y0, y1] += 1

    TOT = np.sum(CNT)

    data_list = []
    for class_i in range(class_num):
        data_list.append([])
        data_list[-1].append(CNT[class_i] / TOT)
        data_list[-1].append(ntu120_class_name[class_i])
        data_list[-1].append(TP[class_i] / CNT[class_i])
        data_list[-1].append([])
        if TP[class_i] == CNT[class_i]:
            continue
        for class_j in range(class_num):
            if FN[class_i][class_j] == 0:
                continue
            data_list[-1][-1].append(
                [FN[class_i][class_j] / max(CNT[class_i] - TP[class_i], 1), ntu120_class_name[class_j]])

        data_list[-1][-1] = sorted(data_list[-1][-1], key=lambda x: x[0], reverse=True)

    data_list = sorted(data_list, key=lambda x: x[2])

    with open(f'{result_file}.csv', 'w') as f:
        f.write(f'Per, Acc, True Class, Wrong Class: ratio, ...\n')
        for data_row in data_list:
            f.write(f'{data_row[0] * 100:.2f}%, {data_row[2] * 100:.2f}%, {data_row[1]}')
            for data_item in data_row[-1]:
                f.write(f', {data_item[1]}: {data_item[0] * 100:.2f}%')
            f.write(f'\n')