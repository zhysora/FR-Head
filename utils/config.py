import argparse
from argparse import ArgumentParser

from torchlight import DictAction


def str2bool(v):
    # 字符串 转 bool
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser: ArgumentParser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')

    # work_dir 是用来储存结果的
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml',
                        help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False,
                        help='if ture, the classification score will be stored')

    # visualize, debug & record
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=32, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for test')

    # classification model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--cl-mode', choices=['ST-Multi-Level'], default=None,
                        help='mode of Contrastive Learning Loss')
    parser.add_argument('--cl-version', choices=['V0', 'V1', 'V2', "NO FN", "NO FP", "NO FN & FP"], default='V0',
                        help='different way to calculate the cl loss')
    parser.add_argument('--pred_threshold', type=float, default=0.0, help='threshold to define the confident sample')
    parser.add_argument('--use_p_map', type=str2bool, default=True,
                        help='whether to add (1 - p_{ik}) to constrain the auxiliary item')
    parser.add_argument('--start-cl-epoch', type=int, default=-1, help='epoch to optimize cl loss')
    parser.add_argument('--w-cl-loss', type=float, default=0.1, help='weight of cl loss')
    parser.add_argument('--w-multi-cl-loss', type=float, default=[0.1, 0.2, 0.5, 1], nargs='+',
                        help='weight of multi-level cl loss')

    # optim
    parser.add_argument('--base-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')  # 一种优化梯度的算法
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser
