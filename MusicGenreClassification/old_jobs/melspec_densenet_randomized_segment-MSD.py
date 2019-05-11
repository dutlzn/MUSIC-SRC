import os
import os.path as osp
import sys
from collections import OrderedDict

sys.path.append(osp.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.msd import MSD_MELSPEC as MSD
from model import DenseInception
from utils import manual_seed
from train_utils import run_epoch, run_epochs


def parse_args():
    parser = argparse.ArgumentParser(description='Hyper-Parameters for training.')
    parser.add_argument('--inference_mode', type=bool, default=False)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_segments', type=int, default=10)
    parser.add_argument('--inference_segments', type=int, default=18)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=0.985)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--save_model_interval', type=int, default=1)
    parser.add_argument('--cuda_visible_devices', type=str, default='0')
    parser.add_argument('--best_model_metric', type=str, default='acc')
    parser.add_argument('--job_name_format', type=str,
                        default='net_bs_{batch_size}_seg_{train_segments}_overlap_{overlap}')
    parser.add_argument('--checkpoint_name_format', type=str,
                        default='checkpoints/MSD/{job_name}/{checkpoint_name}.pt')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='The model checkpoint used when train_mode is False.')
    parser.add_argument('--tb_run_name_format', type=str,
                        default='runs/MSD/{job_name}')
    args = vars(parser.parse_args())
    args['job_name'] = args['job_name_format'].format(**args)
    args['tb_run_name'] = args['tb_run_name_format'].format(**args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda_visible_devices']
    args['device'] = torch.device('cuda:0')
    
    return args


def get_dataloaders(args):
    train_set = MSD(phase='train', min_segments=args['train_segments'],
                    val_size=0.25, test_size=0.25,
                    randomized=True, overlap=args['overlap'])
    val_set = MSD(phase='val', min_segments=args['inference_segments'],
                  val_size=0.25, test_size=0.25, overlap=args['overlap'])
    test_set = MSD(phase='test', min_segments=args['inference_segments'],
                   val_size=0.25, test_size=0.25, overlap=args['overlap'])

    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              shuffle=True, num_workers=args['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            num_workers=args['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             num_workers=args['num_workers'], pin_memory=True)
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    
    return dataloaders


def get_model(args):
    net = DenseInception(1, 32, 3, num_classes=MSD.NUM_CLASSES).to(args['device'])
    criterion = nn.CrossEntropyLoss().to(args['device'])
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args['lr_decay'])
    scheduler.step()
    args['scheduler'] = scheduler
    return net, criterion, optimizer


def main():
    args = parse_args()
    manual_seed(args['random_seed'])
    net, criterion, optimizer = get_model(args)
    dataloaders = get_dataloaders(args)

    if args['inference_mode'] is False:
        tb_writer = SummaryWriter(args['tb_run_name'])
        args['tb_writer'] = tb_writer
        best_net = run_epochs(net, optimizer, dataloaders, criterion, args)
    else:
        net.load_state_dict(torch.load(args['checkpoint']))
        loss, acc = run_epoch(net,
                              optimizer=optimizer, dataloader=dataloaders['test'],
                              criterion=criterion, phase='test', device=args['device'])
        print('Test loss: {:.3f}, Test acc: {:.2%}'.format(loss, acc))


if __name__ == '__main__':
    main()