import copy
import os
import os.path as osp
import pickle
import pprint
from collections import OrderedDict

import numpy as np
import torch
import torch.cuda as cuda
from tqdm import tqdm as tqdm
from sklearn.metrics import roc_auc_score


def manual_seed(seed=0):
    """To reproduce our work, this function set random seeds manually."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def manual_worker_seed(seed=0):
    """To reproduce our work, this function set random seeds manually."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    
def run_extract_features(net, dataloader, device=None):
    net.eval()
    features = []
    labels = []
    with tqdm(dataloader, total=len(dataloader)) as progress:
        for inputs, labels_ in progress:
            num_samples, num_segments, num_channels, num_freqs, num_frames = inputs.shape
            inputs = inputs.type(torch.FloatTensor).to(device)

            with torch.set_grad_enabled(False):
                features_ = net.compute_features(inputs, True)
                features.append(features_)
                labels.append(labels_)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

    
def run_epoch(net, optimizer, dataloader, criterion, phase,
              device=None, with_preds_and_gts=False):
    if dataloader is None:
        return -1, -1
    if phase == 'train':
        net.train()
    else:
        net.eval()

    running_samples = 0
    running_loss = 0
    running_corrects = 0
    predictions = []
    ground_truths = []
    with tqdm(dataloader, total=len(dataloader)) as progress:
        for inputs, labels in progress:
            num_samples, num_segments, num_channels, num_freqs, num_frames = inputs.shape
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                if with_preds_and_gts:
                    predictions.append(outputs.cpu().detach().numpy())
                    ground_truths.append(labels.cpu().detach().numpy())
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_samples += num_samples
            running_loss += loss.item() * num_samples
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

            loss = running_loss / running_samples
            acc = running_corrects / running_samples

            info = OrderedDict(
                phase=phase,
                loss='{:.4f}'.format(loss),
                acc='{:.2%}'.format(acc),
            )

            progress.set_postfix(**info)

    if with_preds_and_gts:
        predictions = np.vstack(predictions)
        ground_truths = np.hstack(ground_truths)
        return loss, acc, predictions, ground_truths
    return loss, acc


def run_epochs(net, optimizer, dataloaders, criterion, args):
    best_net = copy.deepcopy(net)
    best_loss = 1e9
    best_acc = 0
    checkpoint_path = osp.dirname(args['checkpoint_name_format'].format(
        checkpoint_name='placeholder', **args))
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        # configs
        with open(osp.join(checkpoint_path, 'README'), 'w') as f:
            pprint.pprint(args, f, indent=4)
        # Snapshot for code
        src_path = osp.join(checkpoint_path, 'src')
        os.makedirs(src_path)
        backup_cmd = ('find . -type d \( -path ./old_jobs -o -path ./models/deprecated'
                      ' -o -path ./checkpoints \) -prune -o -name "*.py"'
                      ' -exec cp --parents {{}} "{}" \;').format(src_path)
        os.system(backup_cmd)
    patience = args['early_stopping_patience']
    with tqdm(range(args['epochs']), total=args['epochs']) as epoch_progress:
        for epoch in epoch_progress:
            train_loss = 1e9
            train_acc = -1
            val_loss = 1e9
            val_acc = -1
            for phase in ('train', 'val'):
                loss, acc = run_epoch(net, optimizer, dataloaders[phase],
                                      criterion, phase, device=args['device'])

                if phase == 'val':
                    val_loss = loss
                    val_acc = acc
                else:
                    train_loss = loss
                    train_acc = acc
                    if 'scheduler' in args:
                        args['scheduler'].step()
            if val_acc == -1:
                val_loss = train_loss
                val_acc = train_acc
            info = OrderedDict(
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc)
            epoch_progress.set_postfix(**info)

            if 'tb_writer' in args:
                for name, val in info.items():
                    args['tb_writer'].add_scalar(name, val, epoch)         

            if (args['best_model_metric'] == 'acc' and val_acc > best_acc) or (
                args['best_model_metric'] == 'loss' and val_loss < best_loss):
                best_acc = val_acc
                best_loss = val_loss
                best_net.load_state_dict(net.state_dict())
                torch.save(best_net.state_dict(), args['checkpoint_name_format'].format(
                    checkpoint_name='best_model', **args))
                patience = args['early_stopping_patience']
            else:
                patience -= 1
            if patience == 0:
                break
    return best_net


def smooth(x, window_size):
    """Smooth data by sliding window.
    
    Parameters
    ----------
    x: NumPy 1-D array containing the data to be smoothed
    window_size: smoothing window size needs, which must be odd number,
    """
    out0 = np.convolve(x, np.ones(window_size, dtype=int), 'valid') / window_size    
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(x[:window_size-1])[::2] / r
    stop = (np.cumsum(x[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def get_cache_wrapper(cache_path):
    """Return a helper function to materialize output to cache path.
    
    Example
    -------
    class GTZAN(Dataset):
        @utils.get_cache_wrapper('/share/GTZAN/precomputed/vanilla')
        @staticmethod
        def _load_sample(dirname, fname):
            x = features.get_audio(osp.join(dirname, fname))
            return x
    """
    def cache_wrapper(func):
        def cached_func(src_dirname, fname):
            fpath = osp.join(cache_path, fname)
            dirname = osp.dirname(fpath)
            if osp.exists(fpath):
                with open(fpath, 'rb') as f:
                    return pickle.load(f)
            ret = func(src_dirname, fname)
            if not osp.exists(dirname):
                os.makedirs(dirname)
            with open(fpath, 'wb') as f:
                pickle.dump(ret, f)
            return ret
        return cached_func
    return cache_wrapper


def multiclass_roc_auc_score(preds, gts):
    N, M = preds.shape
    aucs = []
    for ith_cls in range(M):
        gts_ = (gts.copy() == ith_cls).astype(np.int32)
        preds_ = preds.copy()[np.arange(N), ith_cls]
        aucs.append(roc_auc_score(gts_, preds_))
    return aucs


raw_print = print
def print(msg):
    raw_print(msg)
    with open('messages.log', 'a+') as f:
        f.write(msg + '\n')