'''
Author: Xiang Pan
Date: 2021-11-22 22:56:22
LastEditTime: 2021-12-16 20:57:23
LastEditors: Xiang Pan
Description: 
FilePath: /project/option.py
@email: xiangpan@nyu.edu
'''
import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',      default='baseline', help='experiment name')
    # parser.add_argument('--batch_size', type=int, default=64,       help='mini-batch size')
    parser.add_argument('--gpu',        type=int, nargs='+',required=False, default = 0)
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--resume_checkpoint_path",   default=None, type=str)
    parser.add_argument("--train_dataset",   default="baseline", type=str)
    parser.add_argument("--net", default="xlnet-base-cased-gat", type=str)
    parser.add_argument("--model_name_or_path", default="xlnet-base-cased")
    parser.add_argument("--num_labels", type=int, default=3)
    
    parser.add_argument("--task_name", default="KITTI")
    parser.add_argument("--log_name", default= None)
    parser.add_argument("--load_checkpoint_path", default=None)
    
    return parser

def get_option():
    parser = get_parser()
    option = parser.parse_args()
    return option