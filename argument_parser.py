import argparse
# import auxiliary.my_utils as my_utils
import os
import datetime
import json
from termcolor import colored
from easydict import EasyDict
from os.path import exists, join


def parser():
    parser = argparse.ArgumentParser()

    # preprocessing

    parser.add_argument('--norm', action="store_true", help='if show_when_test')
    parser.add_argument('--rand_rot', action="store_true", help='if show_when_test')
    parser.add_argument('--rot_z', action="store_true", help='if show_when_test')
    parser.add_argument('--rand_shiver', action="store_true", help='if show_when_test')
    parser.add_argument('--include_normals', action="store_true", help='if show_when_test')
    parser.add_argument('--render_model', action="store_true", help='if show_when_test')
    parser.add_argument('--structure', action="store_true", help='if show_when_test')
    parser.add_argument('--circular_conv', action="store_true", help='if show_when_test')
    parser.add_argument('--local_frame', action="store_true", help='if show_when_test')
    parser.add_argument('--positional_encoder', action="store_true", help='if show_when_test')
    parser.add_argument('--with_norm', action="store_true", help='if show_when_test')
    parser.add_argument('--fc', action="store_true", help='if show_when_test')
    parser.add_argument('--decomp_conv', action="store_true", help='if show_when_test')
    parser.add_argument('--atten', action="store_true", help='if show_when_test')
    parser.add_argument('--with_embedding', action="store_true", help='if show_when_test')
    parser.add_argument('--with_curv', action="store_true", help='if show_when_test')
    parser.add_argument('--adjust_normal', action="store_true", help='if show_when_test')

    parser.add_argument('--jitter', type=float, default=0., help='if show_when_test')

    # ablation
    parser.add_argument('--lrf', action="store_true", help='if show_when_test')
    parser.add_argument('--loc_only', action="store_true", help='if show_when_test')
    parser.add_argument('--ppf_only', action="store_true", help='if show_when_test')
    parser.add_argument('--no_edge', action="store_true", help='if show_when_test')

    parser.add_argument('--no_normal', action="store_true", help='if show_when_test')
    parser.add_argument('--OP_as_normal', action="store_true", help='if show_when_test')

    # ablation
    parser.add_argument('--vis_feat_map', action="store_true", help='if show_when_test')

    # robustness
    parser.add_argument('--o3d_normal', action="store_true", help='if show_when_test')
    parser.add_argument('--o3d_normal_orien', action="store_true", help='if show_when_test')
    parser.add_argument('--density', action="store_true", help='if show_when_test')

    parser.add_argument('--axes', nargs="+", default=None, help='if show_when_test')

    parser.add_argument('--init_attr', type=str, default='default', help='if show_when_test')


    # Train Mode
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--use_sgd", action="store_true")
    parser.add_argument("--use_step", action="store_true")
    parser.add_argument("--use_annl", action="store_true")

    parser.add_argument("--voting", action="store_true")
    parser.add_argument("--vote_num", type=int, default=12)

    parser.add_argument("--task", type=str, default='cls')

    # Training parameters
    parser.add_argument("--network", type=str, default='GRNet', help="if test while training")

    parser.add_argument("--training", type=int, default=1)
    parser.add_argument("--test", type=bool, default=False, help="if test while training")
    parser.add_argument("--test_final", type=bool, default=False, help="if test while training")

    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--lr_schedule', type=list, default=[], help='')
    parser.add_argument('--step_size', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--gamma', type=float, default=0.5, help='for step_LR')
    parser.add_argument('--dp_rate', type=float, default=0.5, help='for step_LR')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='for step_LR')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='for step_LR')

    parser.add_argument('--warm_up', type=int, default=-1, help='for step_LR')
    parser.add_argument('--tune_AE', type=int, default=-1, help='for step_LR')

    parser.add_argument('--sample_points', type=int, default=2048, help='number of epochs to train for')

    parser.add_argument('--margin', type=float, default=5., help='')

    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=10, help='input batch size')

    parser.add_argument('--show_when_test', action="store_true", help='if show_when_test')

    parser.add_argument('--workers', type=int, help='number of data loading workers',
                        default=16)  # loading time < training time, loading faster cannot improve efficiency

    parser.add_argument('--dataset', type=str, default='MCB', help='Faust path')

    parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--random_translate", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
                        help="apply data augmentation : axial rotation ")
    parser.add_argument('--checkpointname', type=str, default="", help='name of checkpoint.')
    parser.add_argument('--not_load_dict', action="store_true", help='if load checkpoint.')
    parser.add_argument('--dir_name', type=str, default="/data1/crh/proj/3DGR/log", help='name of the log folder.')
    parser.add_argument('--visdom_port', type=int, default=8890, help="visdom port")
    parser.add_argument('--env', type=str, default='MPNet', help='visdom environment')

    #Loss parameters
    parser.add_argument('--w1', type=float, default=1., help='weight for classification loss')
    parser.add_argument('--w2', type=float, default=0.1, help='weight for classification loss')

    parser.add_argument('--MMD', action="store_true", help='weight for classification loss')

    parser.add_argument('--weight_sample_loss', type=float, default=10, help='weight for classification loss')
    parser.add_argument('--weight_temperature_loss', type=float, default=1., help='weight for classification loss')
    parser.add_argument('--weight_EI_loss', type=float, default=0.1, help='weight for classification loss')

    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--tcl", action="store_true")
    parser.add_argument("--stn", action="store_true")
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # -----------------------------------------------------------------------------------------------------
    parser.add_argument('--split', type=str, default="main", help='training name')


    # -----------------------------------------------------------------------------------------------------

    parser.add_argument('--id', type=str, default="0", help='training name')
    # parser.add_argument('--env', type=str, default="Atlasnet", help='visdom environment')
    # parser.add_argument('--http_port', type=int, default=8891, help="http port")
    parser.add_argument('--demo_input_path', type=str, default="./doc/pictures/plane_input_demo.png", help='dirname')
    parser.add_argument('--reload_decoder_path', type=str, default="", help='dirname')
    parser.add_argument('--reload_model_path', type=str, default='', help='optional reload model path')
    parser.add_argument('--model_path', type=str, default='', help='optional reload model path')

    # Network
    parser.add_argument('--GNN_layers', type=list, default=['self', 'cross'] * 3,
                        help='number of hidden SuperGraph Layer')
    parser.add_argument('--retrieve', action="store_true", help='dirname')
    parser.add_argument('--k', type=int, default=20, help='dirname')


    parser.add_argument('--differentiable_sample_strategy', type=str, help='differentiable_sample_strategy')


    opt = parser.parse_args()

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    opt = EasyDict(opt.__dict__)

    opt.logname = join(opt.dir_name, 'log.txt')



    return opt
