import os.path as osp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm as tqdm
from datasets.msd import MSD_MELSPEC as MSD
from datasets.features import get_melspectrogram


def check_valid(fname):
    try:
        x = np.load(osp.join('/share',
                             'MSD',
                             'precomputed/vanilla/{}.npy'.format(fname)))
        if x.shape[0] < 660980:
            raise Exception("Too short")
        x = get_melspectrogram(x)
    except Exception as e:
        print(fname, e)
        return False
    return True


dataset = MSD()
chunksize=10
with ProcessPoolExecutor() as executor:
    futures = executor.map(check_valid, dataset.X, chunksize=chunksize)
    for _ in tqdm(futures, total=len(dataset.X)):
        pass
