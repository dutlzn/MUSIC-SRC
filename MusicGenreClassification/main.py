import os
import os.path as osp
import sys
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from args import (
    parser, build_args, build_dataset_class, build_model_class,
    build_loss_func_class, build_randomized_method)
from utils import manual_seed, manual_worker_seed, run_epoch, run_epochs, multiclass_roc_auc_score


def get_dataloaders(args):
    Dataset = build_dataset_class(args)
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
    
    return dataloaders


def get_model(args):
    Model = build_model_class(args)
    Dataset = build_dataset_class(args)
    Loss = build_loss_func_class(args)
    criterion = Loss().to(args['device'])
    net = Model(
        1,
        growth_rate=args['growth_rate'],
        block_config=args['block_config'],
        num_init_features=args['num_init_features'],
        bn_size=args['bn_size'],
        drop_rate=args['drop_rate'],
        num_classes=Dataset.NUM_CLASSES).to(args['device'])
    if args['fine_tuning_msd_checkpoint'] != '':
        import models.v2 as model_zoo_v2
        from datasets.msd import MSD_MELSPEC as MSD
        msd_net = model_zoo_v2.DCIN_V2(1, num_classes=MSD.NUM_CLASSES)
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
    args = build_args()
    manual_seed(args['random_seed'])
    net, criterion, optimizer = get_model(args)
    dataloaders = get_dataloaders(args)

    if args['inference_mode'] is False:
        tb_writer = SummaryWriter(args['tb_run_name'])
        args['tb_writer'] = tb_writer
        run_epochs(net, optimizer, dataloaders, criterion, args)
    if args['checkpoint'] == '':
        args['checkpoint'] = args['checkpoint_name_format'].format(
            checkpoint_name='best_model', **args)
    net.load_state_dict(torch.load(args['checkpoint']))
    loss, acc, preds, gts = run_epoch(
        net, optimizer=optimizer, dataloader=dataloaders['test'],
        criterion=criterion, phase='test', device=args['device'],
        with_preds_and_gts=True)
    aucs = multiclass_roc_auc_score(preds, gts)
    print('{} Test loss: {:.3f}, Test acc: {:.2%}, Test AUC: {}'.format(
        args['job_name'], loss, acc, aucs))


if __name__ == '__main__':
    main()