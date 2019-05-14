import os
import os.path as osp
import sys
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from args import (
    parser, build_args, build_dataset_class, build_model_class,
    build_loss_func_class, build_randomized_method, build_block_class)
from utils import manual_seed, manual_worker_seed, run_epoch, run_epochs, multiclass_roc_auc_score, print

'''
类似这样输入命令
python kfold_main.py --model DCIN --block_config 3 3 --drop_rate 0.2 --dataset GTZAN --train_model
'''

def get_dataloaders(args):
    Dataset = build_dataset_class(args)
    all_set = Dataset(phase='all', root=args['dataset_root'])
    train_set = Dataset(phase='train', min_segments=args['train_segments'],
                        randomized_method=build_randomized_method(args),
                        random_seed=args['random_seed'],
                        val_size=args['val_size'], test_size=args['test_size'],
                        overlap=args['overlap'], noise_rate=args['noise_rate'],
                        root=args['dataset_root'])
    val_set = Dataset(phase='val', min_segments=args['inference_segments'],
                      random_seed=args['random_seed'],
                      val_size=args['val_size'], test_size=args['test_size'],
                      overlap=args['overlap'], root=args['dataset_root'])
    test_set = Dataset(phase='test', min_segments=args['inference_segments'],
                       random_seed=args['random_seed'],
                       val_size=args['val_size'], test_size=args['test_size'],
                       overlap=args['overlap'], root=args['dataset_root'])
    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              shuffle=True, num_workers=args['num_workers'],
                              pin_memory=True, worker_init_fn=manual_worker_seed)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'],
                            num_workers=args['num_workers'], pin_memory=True,
                            worker_init_fn=manual_worker_seed)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'],
                             num_workers=args['num_workers'], pin_memory=True,
                             worker_init_fn=manual_worker_seed)
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    skf = StratifiedKFold(n_splits=args['num_kfold'],
                          shuffle=True, random_state=args['random_seed'])
    for train_index, test_index in skf.split(all_set.X, all_set.Y):
        train_index, val_index = train_test_split(
            train_index, test_size=args['val_size'],
            random_state=args['random_seed'], stratify=all_set.Y[train_index])
        train_set.X = all_set.X[train_index]
        train_set.Y = all_set.Y[train_index]
        val_set.X = all_set.X[val_index]
        val_set.Y = all_set.Y[val_index]
        test_set.X = all_set.X[test_index]
        test_set.Y = all_set.Y[test_index]
        
        yield dataloaders


def get_model(args):
    def _make_model_params(args):
        Block = build_block_class(args)
        return {
            'growth_rate': args['growth_rate'],
            'block_config': args['block_config'],
            'num_init_features': args['num_init_features'],
            'bn_size': args['bn_size'],
            'drop_rate': args['drop_rate'],
            'Block': Block,
        }
    
    Model = build_model_class(args)
    Dataset = build_dataset_class(args)
    Loss = build_loss_func_class(args)
    criterion = Loss().to(args['device'])
    params = _make_model_params(args)
    net = Model(1, num_classes=Dataset.NUM_CLASSES, **params).to(args['device'])
    if args['fine_tuning_msd_checkpoint'] != '':
        from datasets.msd import MSD_MELSPEC as MSD
        
        msd_net = Model(1, num_classes=MSD.NUM_CLASSES, **params)
        msd_net.load_state_dict(torch.load(args['fine_tuning_msd_checkpoint']))
        net.features.load_state_dict(msd_net.features.state_dict())
        del msd_net
        params = [{'params': m.parameters()} for name, m in net.named_children() if name != 'features']
        params += [{'params': net.features.parameters(), 'lr': args['lr'] / 100}]
        optimizer = optim.Adam(params, lr=args['lr'], weight_decay=args['weight_decay'])
    else:
        optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args['lr_decay'])
    scheduler.step()
    args['scheduler'] = scheduler
    return net, criterion, optimizer


def main():
    raw_args = build_args()
    manual_seed(raw_args['random_seed'])
    original_job_name_suffix = raw_args['job_name_suffix']
    raw_args['append_timestamp_to_job_name'] = False
    accs = []
    for kfold, dataloaders in enumerate(get_dataloaders(raw_args)):
        raw_args['job_name_suffix'] = '{}_kfold_{}'.format(original_job_name_suffix, kfold)
        args = build_args(raw_args)
        
        if kfold < args['kfold_start']:
            continue
        net, criterion, optimizer = get_model(args)

        # Train
        if args['inference_mode'] is False:
            tb_writer = SummaryWriter(args['tb_run_name'])
            args['tb_writer'] = tb_writer
            run_epochs(net, optimizer, dataloaders, criterion, args)
        
        # Evaluate
        args['checkpoint'] = args['checkpoint_name_format'].format(
            checkpoint_name='best_model', **args)
        net.load_state_dict(torch.load(args['checkpoint']))
        loss, acc, preds, gts = run_epoch(
            net, optimizer=optimizer, dataloader=dataloaders['test'],
            criterion=criterion, phase='test', device=args['device'],
            with_preds_and_gts=True)
        aucs = multiclass_roc_auc_score(preds, gts)
        print('Fold {}: {} Test loss: {:.3f}, Test acc: {:.2%}, Test AUC: {}'.format(
            kfold, args['job_name'], loss, acc, aucs))
        accs.append(acc)
    print('Average test acc: {:.2%}'.format(np.mean(accs)))


if __name__ == '__main__':
    main()