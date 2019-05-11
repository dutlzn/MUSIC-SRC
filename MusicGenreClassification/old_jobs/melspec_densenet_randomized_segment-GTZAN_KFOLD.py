import os
import os.path as osp
import sys
from collections import OrderedDict

sys.path.append(osp.abspath('..'))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.gtzan import GTZAN_MELSPEC as GTZAN
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
    parser.add_argument('--num_kfold', type=int, default=10)
    parser.add_argument('--save_model_interval', type=int, default=1)
    parser.add_argument('--cuda_visible_devices', type=str, default='0')
    parser.add_argument('--best_model_metric', type=str, default='acc')
    parser.add_argument('--job_name_format', type=str,
                        default='net_bs_{batch_size}_seg_{train_segments}_overlap_{overlap}')
    parser.add_argument('--checkpoint_name_suffix', type=str, default='')
    parser.add_argument(
        '--checkpoint_name_format', type=str,
        default='checkpoints/GTZAN_KFOLD/{job_name}/{checkpoint_name}{checkpoint_name_suffix}.pt')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='The model checkpoint used when train_mode is False.')
    parser.add_argument('--tb_run_name_format', type=str,
                        default='runs/GTZAN_KFOLD/{job_name}')
    args = vars(parser.parse_args())
    args['job_name'] = args['job_name_format'].format(**args)
    args['tb_run_name'] = args['tb_run_name_format'].format(**args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args['cuda_visible_devices']
    args['device'] = torch.device('cuda:0')
    
    return args


def get_datasets(args):
    dataset = GTZAN(phase='all', min_segments=args['train_segments'])
    train_set = GTZAN(phase='all', min_segments=args['train_segments'],
                      randomized=True, overlap=args['overlap'])
    val_set = GTZAN(phase='all', min_segments=args['inference_segments'],
                    overlap=args['overlap'])
    test_set = GTZAN(phase='all', min_segments=args['inference_segments'],
                     overlap=args['overlap'])
    
    return {
        'all': dataset,
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }


def get_dataloaders(datasets, args):
    train_loader = None
    val_loader = None
    test_loader = None
    if 'train' in datasets:
        train_loader = DataLoader(
            datasets['train'], batch_size=args['batch_size'],
            shuffle=True, num_workers=args['num_workers'],
            pin_memory=True)
    if 'val' in datasets:
        val_loader = DataLoader(
            datasets['val'], batch_size=args['batch_size'],
            num_workers=args['num_workers'], pin_memory=True)
    if 'test' in datasets:
        test_loader = DataLoader(
            datasets['test'], batch_size=args['batch_size'],
            num_workers=args['num_workers'], pin_memory=True)
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }


def get_model(args):
    net = DenseInception(1, 32, 3, num_classes=ISMIR.NUM_CLASSES).to(args['device'])
    criterion = nn.CrossEntropyLoss().to(args['device'])
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args['lr_decay'])
    scheduler.step()
    args['scheduler'] = scheduler
    return net, criterion, optimizer


def set_cv_indices(datasets, train_index, test_index):
    whole_dataset = datasets['all']
    train_set = datasets['train']
    val_set = datasets['val']
    test_set = datasets['test']
    
    train_index, val_index = train_test_split(
        train_index, test_size=1 / 9,
        random_state=args['random_seed'],
        stratify=whole_dataset.Y[train_index])

    train_set.X, train_set.Y = whole_dataset.X[train_index], whole_dataset.Y[train_index]
    val_set.X, val_set.Y = whole_dataset.X[val_index], whole_dataset.Y[val_index]
    test_set.X, test_set.Y = whole_dataset.X[test_index], whole_dataset.Y[test_index]


def run_kfold(args):
    datasets = get_datasets(args)
    cv_results = []
    skf = StratifiedKFold(args['num_kfold'], shuffle=True, random_state=args['random_seed'])
    for kfold, (train_index, test_index) in enumerate(
        skf.split(whole_dataset.X, whole_dataset.Y)):
        
        set_cv_indices(datasets, train_index, test_index)
        dataloaders = get_dataloaders(datasets, args)

        net, criterion, optimizer = get_model(args)
        args['tb_writer'] = SummaryWriter(args['tb_run_name'] + '_Fold_{}'.format(kfold))
        args['checkpoint_name_suffix'] = '_Fold_{}'.format(kfold)
        best_net = run_epochs(net, optimizer, dataloaders, criterion, args)
        
        fold_info = {}
        for dataset_phase in ('train', 'val', 'test'):
            loss, acc = run_epoch(
                best_net,
                optimizer=optimizer,
                dataloader=dataloaders[dataset_phase],
                criterion=criterion,
                phase='test',
                device=args['device'])
            fold_info['{}_loss'.format(dataset_phase)] = loss
            fold_info['{}_acc'.format(dataset_phase)] = acc
        cv_results.append(fold_info)
        
    print('\n')
    for kfold, result in enumerate(cv_results):
        print('Fold {}, '
              'train loss: {train_loss:.4f}, train acc: {train_acc:.2%}, '
              'val loss: {val_loss:.4f}, val acc: {val_acc:.2%}, '
              'test loss: {test_loss:.4f}, test acc: {test_acc:.2%}'.format(
            kfold, **result))

    print('{}-fold cross-validation'.format(len(cv_results)))
    for phase in ('train', 'val', 'test'):
        print('{} loss: {:.4f}'.format(
            phase,
            sum(x['{}_loss'.format(phase)] for x in cv_results) / len(cv_results)))
        print('{} acc: {:.2%}'.format(
            phase,
            sum(x['{}_acc'.format(phase)] for x in cv_results) / len(cv_results)))


def main():
    args = parse_args()
    manual_seed(args['random_seed'])
    run_kfold(args)


if __name__ == '__main__':
    main()