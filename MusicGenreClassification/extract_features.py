import os
import os.path as osp

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from args import build_args
from utils import manual_seed, run_extract_features


def get_dataloaders(args):
    Dataset = args['dataset']
    train_set = Dataset(phase='train', min_segments=args['inference_segments'],
                        random_seed=args['random_seed'],
                        overlap=args['overlap'], root=args['dataset_root'])
    val_set = Dataset(phase='val', min_segments=args['inference_segments'],
                      random_seed=args['random_seed'],
                      overlap=args['overlap'], root=args['dataset_root'])
    test_set = Dataset(phase='test', min_segments=args['inference_segments'],
                       random_seed=args['random_seed'],
                       overlap=args['overlap'], root=args['dataset_root'])

    train_loader = DataLoader(train_set, batch_size=args['batch_size'],
                              num_workers=args['num_workers'], pin_memory=True)
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
    Model = args['model']
    Dataset = args['dataset']
    net = Model(
        1, block_config=args['block_config'],
        num_classes=Dataset.NUM_CLASSES, drop_rate=args['drop_rate']).to(args['device'])
    return net


def main():
    args = build_args()
    assert args['inference_segments'] == 1
    manual_seed(args['random_seed'])
    net = get_model(args)
    dataloaders = get_dataloaders(args)

    net.load_state_dict(torch.load(args['checkpoint']))
    for phase, dataloader in dataloaders.items():
        features, labels = run_extract_features(net, dataloader, device=args['device'])
        features = torch.squeeze(features).cpu().numpy()
        labels = labels.cpu().numpy()
        df = pd.concat([pd.DataFrame(features), pd.DataFrame(labels)], axis=1)
        fpath = osp.join(args['output_path'], '{job_name}_{phase}.csv'.format(phase=phase, **args))
        if not osp.exists(osp.dirname(fpath)):
            os.makedirs(osp.dirname(fpath))
        df.to_csv(fpath, header=False, index=False)

if __name__ == '__main__':
    main()