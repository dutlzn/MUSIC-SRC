import os.path as osp

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import datasets.features as features
import datasets.utils as datasets_utils


class MSD(datasets_utils.Dataset):
    """A dataset with raw Million Song Dataset audio clips."""
    NUM_CLASSES = 13
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        X, Y = self._load_data(self.root)
        phase = self.phase
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

    def _load_sample(self, fname):
        return np.load(osp.join(self.root, 'MSD', 'precomputed/vanilla/{}.npy'.format(fname)))

    def _load_data(self, cache_path='/share/MSD/precomputed/vanilla/X.npy'):
        root = self.root
        df = pd.read_csv(osp.join(root, 'MSD', 'dataset.csv'))
        X = df['track_id'].tolist()
        X = [osp.join('tracks', '{}.mp3'.format(x)) for x in X]
        Y = np.array(df['genre'].tolist())

        return X, Y

    def __getitem__(self, idx):
        x = self._load_sample(self.X[idx])
        if self.transforms:
            x = self.transforms(x)
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.X)
    
    
class MSD_MELSPEC(datasets_utils.FeatureExtracterDataSet,
                  datasets_utils.SegmentDataset,
                  datasets_utils.NoisedDataSet,
                  MSD):
    """MSD dataset with melspectrogram as x."""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_extractor=features.get_melspectrogram, *args, **kwargs)
