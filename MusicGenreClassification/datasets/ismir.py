import os.path as osp
from collections import Counter

import pandas as pd
import numpy as np

import datasets.features as features
import datasets.utils as datasets_utils


class ISMIR2004(datasets_utils.Dataset):
    """A dataset with raw ISMIR 2004 Dataset audio clips."""
    NUM_CLASSES = 6
    _LABEL_MAPPING = None
    _CLASS_WEIGHT = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        phase = self.phase
        self.phase_map = {
            'train': 'training',
            'val': 'development',
            'test': 'evaluation',
            'all': 'all',
        }
        self._phase = self.phase_map[phase]
        
        X, Y = self._load_data()
        self.X = X
        self.Y = Y

    def _load_sample(self, fname):
        fpath = osp.join(self.root, 'ISMIR2004', 'precomputed/vanilla', '{}.npy'.format(fname))
        return np.load(fpath)

    @staticmethod
    def _load_dataframe(phase, path):
        df = pd.read_csv(path, header=None)[[0, 5]]
        df = df.rename(index=str, columns={0: 'genre', 5: 'path'})
        df['path'] = df['path'].map(lambda x: osp.join(phase, x))
        return df
    
    @staticmethod
    def _make_metadata_path(root, phase):
        return osp.join(root, 'ISMIR2004', 'metadata', phase, 'tracklist.csv')

    def _load_phase_dataframe(self):
        if self.phase == 'all':
            return pd.concat([self._load_dataframe(phase, self._make_metadata_path(self.root, phase))
                              for _, phase in self.phase_map.items() if phase != 'all'],
                             axis=0, ignore_index=True)
        else:
            return self._load_dataframe(self._phase, self._make_metadata_path(self.root, self._phase))

    def _load_data(self):
        df = self._load_phase_dataframe()
        X = df['path'].tolist()
        Y = df['genre'].tolist()
        if self._LABEL_MAPPING is None:
            sorted_Y = sorted(list(set(Y)))
            type(self)._LABEL_MAPPING = {x: idx for x, idx in zip(sorted_Y, range(len(sorted_Y)))}
        Y = [self._LABEL_MAPPING[y] for y in Y]
        if self._CLASS_WEIGHT is None:
            counter = Counter(Y)
            type(self)._CLASS_WEIGHT = {label: (len(Y) - cnt) / len(Y)
                                        for label, cnt in counter.items()}
        return np.array(X), np.array(Y)

    def __getitem__(self, idx):
        x = self._load_sample(self.X[idx])
        if self.transforms:
            x = self.transforms(x)
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.X)
    
    
class ISMIR2004_MELSPEC(datasets_utils.FeatureExtracterDataSet,
                        datasets_utils.SegmentDataset,
                        datasets_utils.NoisedDataSet,
                        ISMIR2004):
    """ISMIR2004 dataset with melspectrogram as x."""
    def __init__(self, *args, **kwargs):
        super().__init__(feature_extractor=features.get_melspectrogram, *args, **kwargs)
