import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
'''
类似这样输入命令
python kfold_main.py --model DCIN --block_config 3 3 --drop_rate 0.2 --dataset GTZAN --train_model
'''
parser = argparse.ArgumentParser(
    description='Parameters for the training and the model.')

# Train & Inference settings
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--inference_mode', dest='inference_mode',
                    action='store_true', default=True)
parser.add_argument('--train_mode', dest='inference_mode', action='store_false')
parser.add_argument('--inference_segments', type=int, default=18)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.985)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--overlap', type=float, default=0.5)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--train_segments', type=int, default=4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--early_stopping_patience', type=int, default=80)

# Model & Dataset settings
parser.add_argument('--block_type', type=str, default='DenseBlock')
parser.add_argument('--block_config', type=int, nargs='+', default=[3, 3, 6, 4])
parser.add_argument('--dataset', type=str, default='ISMIR2004')
parser.add_argument('--dataset_root', type=str, default='/share')
parser.add_argument('--growth_rate', type=int, default=32)
parser.add_argument('--bn_size', type=int, default=4)
parser.add_argument('--num_init_features', type=int, default=64)
parser.add_argument('--drop_rate', type=float, default=0)
parser.add_argument('--fine_tuning_msd_checkpoint', type=str, default='')
parser.add_argument('--loss_func', type=str, default='NLLLoss')
parser.add_argument('--model', type=str, default='DCIN')
parser.add_argument('--noise_rate', type=float, default=0)
parser.add_argument('--num_kfold', type=int, default=10)
parser.add_argument('--kfold_start', type=int, default=0)
parser.add_argument('--randomized_method', type=str, default='sequential_random')
parser.add_argument('--test_size', type=float, default=0.1)
parser.add_argument('--val_size', type=float, default=0.1)

# Other
parser.add_argument('--best_model_metric', type=str, default='acc')
parser.add_argument('--checkpoint_name_format', type=str,
                    default='checkpoints/{job_name}/{checkpoint_name}.pt')
parser.add_argument('--checkpoint', type=str, default='',
                    help='The model checkpoint used when train_mode is False.')
parser.add_argument('--cuda_visible_devices', type=str, default='0')
parser.add_argument('--job_name_format', type=str,
                    default='{dataset}_{model}_block_config_{block_config}_drop_rate_{drop_rate}{job_name_suffix}')
parser.add_argument('--tb_run_name_format', type=str,
                    default='runs/{job_name}')
parser.add_argument('--job_name_suffix', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--append_timestamp_to_job_name', dest='append_timestamp_to_job_name',
                    action='store_true', default=True)
parser.add_argument('--no_timestamp_to_job_name', dest='append_timestamp_to_job_name',
                    action='store_false')
parser.add_argument('--timestamp', type=str, default=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))


def build_model_class(args):
    import models.v2 as model_zoo_v2

    models = {
        'DCIN': model_zoo_v2.DCIN,
        'DCIN_v2': model_zoo_v2.DCIN_V2,
        'WDCIN': model_zoo_v2.WDCIN,
        'CF_DCIN': model_zoo_v2.ContextFreeDCIN,
        'CFF_DCIN': model_zoo_v2.ContextFreeFeatureDCIN,
        'CR_DCIN': model_zoo_v2.ContextRelatedDCIN,
    }
    
    return models[args['model']]


def build_block_class(args):
    import models.basic as basic_models
    
    blocks = {
        'DenseBlock': basic_models.DenseBlock,
        'NativeDenseBlock': basic_models.NativeDenseBlock,
    }
    
    return blocks[args['block_type']]

def build_dataset_class(args):
    from datasets.ismir import ISMIR2004_MELSPEC as ISMIR
    from datasets.gtzan import GTZAN_MELSPEC as GTZAN
    from datasets.gtzan import GTZANFF_MELSPEC as GTZANFF
    from datasets.msd import MSD_MELSPEC as MSD

    
    datasets = {
        'ISMIR2004': ISMIR,
        'GTZAN': GTZAN,
        'GTZANFF': GTZANFF,
        'MSD': MSD,
    }
    
    return datasets[args['dataset']]


def build_loss_func_class(args):
    losses = {
        'NLLLoss': nn.NLLLoss,
    }
    
    return losses[args['loss_func']]


def build_randomized_method(args):
    from datasets.utils import SegmentDataset
    
    randomized_methods = {
        'no_random': SegmentDataset.NONE,
        'sequential_random': SegmentDataset.SEQUENTIAL_RANDOM,
        'nonsequential_random': SegmentDataset.NONSEQUENTIAL_RANDOM,
    }
    
    return randomized_methods[args['randomized_method']]


def build_args(raw_args=None):
    if raw_args is None:
        args = vars(parser.parse_args())
    else:
        args = raw_args
    if args['append_timestamp_to_job_name'] is True:
        args['job_name_suffix'] = '_{}{}'.format(
            args['timestamp'], args['job_name_suffix'])
    args['job_name'] = args['job_name_format'].format(**args)
    args['tb_run_name'] = args['tb_run_name_format'].format(**args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda_visible_devices']
    args['device'] = torch.device('cuda:0')
    return args


if __name__ == '__main__':
    print(build_args())