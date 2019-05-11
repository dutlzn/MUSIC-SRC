import os.path as osp

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import datasets.features as features
import datasets.utils as datasets_utils


class GTZAN(datasets_utils.Dataset):
    """A dataset with raw GTZAN audio clips."""
    NUM_CLASSES = 10
    _INDICES = None
    _X = None
    _Y = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        phase = self.phase
        INDICES, X, Y = self._load_data()
        if phase not in ('train', 'val', 'test'):
            self.X = X
            self.Y = Y
            return
        
        train_x, val_x, test_x, train_y, val_y, test_y = self._split(X, Y)

        if phase == 'train':
            self.X = train_x
            self.Y = train_y
        elif phase == 'val':
            self.X = val_x
            self.Y = val_y
        elif phase == 'test':
            self.X = test_x
            self.Y = test_y
    
    def _split(self, X, Y):
        train_x, val_test_x, train_y, val_test_y = train_test_split(
            X, Y, test_size=self.test_size + self.val_size,
            random_state=self.random_seed, stratify=Y)  

        if self.val_size == 0:
            val_x = []
            val_y = []
            test_x = val_test_x
            test_y = val_test_y
        else:
            val_x, test_x, val_y, test_y = train_test_split(
                val_test_x, val_test_y, test_size=self.test_size / (self.test_size + self.val_size),
                random_state=self.random_seed, stratify=val_test_y)
        return train_x, val_x, test_x, train_y, val_y, test_y

    def _load_sample(self, dirname, fname):
        return features.get_audio(osp.join(dirname, fname))

    def _load_data(self, cache_path='/share/GTZAN/precomputed/vanilla/X.npy'):
        if self._X is not None:
            return self._INDICES, self._X, self._Y
        
        root = osp.join(self.root, 'GTZAN')
        metadata_file_path = osp.join(root, 'files.csv')
        X = []
        Y = []
        
        # Read file paths from metadata file
        with open(metadata_file_path, 'r') as f:
            for line in f:
                fname, fpath, fcate = line.strip().split(' ')
                X.append(fpath)
                Y.append(fcate)
                
        # Map string categories to int categories
        all_cates = sorted(list(set(Y)))
        mapping = {x: idx for x, idx in zip(all_cates, range(len(all_cates)))}
        Y = np.array([mapping[y] for y in Y])
        
        INDICES = {x: idx for x, idx in zip(X, range(len(X)))}
        # Load raw audios
        if osp.exists(cache_path):
            with open(cache_path, 'rb') as f:
                X = np.load(f)
        else:
            X = np.array([self._load_sample(root, fname) for fname in X])
            with open(cache_path, 'wb') as f:
                np.save(f, X)
                
        type(self)._INDICES = INDICES
        type(self)._X = X
        type(self)._Y = Y
        return INDICES, X, Y

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transforms:
            x = self.transforms(x)
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.X)

    
class GTZANFF(GTZAN):
    """GTZAN dataset(fault-filtered version).
    
    See paper: Deep Learning and Music Adversaries
    """
    def _split(self, X, Y):
        indices = {}
        for phase, list_path in (('train', 'filtered_train_files.txt'),
                                 ('val', 'filtered_val_files.txt'),
                                 ('test', 'filtered_test_files.txt')):
            with open(osp.join(self.root, 'GTZAN', list_path), 'r') as f:
                fpaths = []
                for fpath in f:
                    fname, fext = fpath.rsplit('.', 1)
                    fpath = 'genres/{}.au'.format(fname)
                    fpaths.append(fpath)
            indices[phase] = [self._INDICES[fpath] for fpath in fpaths]
        train_x = X[indices['train']]
        val_x = X[indices['val']]
        test_x = X[indices['test']]
        train_y = Y[indices['train']]
        val_y = Y[indices['val']]
        test_y = Y[indices['test']]
        return train_x, val_x, test_x, train_y, val_y, test_y
    

class GTZAN_SPEC(datasets_utils.FeatureExtracterDataSet,
                 datasets_utils.SegmentDataset,
                 datasets_utils.NoisedDataSet,
                 GTZAN):
    """GTZAN dataset with spectrogram as x."""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_extractor=features.get_spectrogram, *args, **kwargs)

    
class GTZAN_MELSPEC(datasets_utils.FeatureExtracterDataSet,
                    datasets_utils.SegmentDataset,
                    datasets_utils.NoisedDataSet,
                    GTZAN):
    """GTZAN dataset with melspectrogram as x."""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_extractor=features.get_melspectrogram, *args, **kwargs)

    
class GTZAN_MFCC(datasets_utils.FeatureExtracterDataSet,
                 datasets_utils.SegmentDataset,
                 datasets_utils.NoisedDataSet,
                 GTZAN):
    """GTZAN dataset with MFCC as x."""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_extractor=features.get_MFCC, *args, **kwargs)

        
class GTZANFF_MELSPEC(datasets_utils.FeatureExtracterDataSet,
                      datasets_utils.SegmentDataset,
                      datasets_utils.NoisedDataSet,
                      GTZANFF):
    """GTZAN dataset with melspectrogram as x."""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_extractor=features.get_melspectrogram, *args, **kwargs)