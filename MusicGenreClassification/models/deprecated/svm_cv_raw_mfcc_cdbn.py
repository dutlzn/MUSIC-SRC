import os
import os.path as osp
import sys
import pickle
from itertools import combinations
sys.path.append(osp.abspath('..'))

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from datasets.gtzan import GTZAN_SPEC, GTZAN_MFCC


def populate_features(src_dataset, dst_x, dst_y, num_segments):
    for sample_idx, (x, y) in tqdm(enumerate(src_dataset), total=len(src_dataset), leave=False):
        for segment_idx, x_ in enumerate(x):
            mean = np.mean(x_, axis=1)
            max = np.max(x_, axis=1)
            std = np.std(x_, axis=1)
            idx = sample_idx * num_segments + segment_idx
            dst_x[idx:idx+1] = np.hstack((mean, max, std))
            dst_y[idx] = y

            
def populate_cdbn_features(src_dataset, dst_x, dst_y, num_segments,
                           cdbn_checkpoint='cdbn_checkpoints/checkpoint_layer_0_epoch_90.pt'):
    Ewhiten = None
    with open('pca_whiten_mat.npy', 'rb') as f:
        Ewhiten = np.load(f)
    assert Ewhiten is not None
    with open(cdbn_checkpoint, 'rb') as f:
        cdbn = torch.load(f)

    for sample_idx, (x, y) in tqdm(enumerate(src_dataset), total=len(src_dataset), leave=False):
        for segment_idx, x_ in enumerate(x):
            x_ = Ewhiten.dot(x_)
            x_ = torch.from_numpy(x_)
            x_ = x_.type(torch.FloatTensor)
            x_ = x_[None, None, :, 1:]
            x_ = x_.to(0)
            x_ = cdbn.crbms[0].v2h(x_)[1].squeeze_().cpu().numpy()
            mean = []
            max = []
            std = []
            mean = np.mean(x_, axis=1)
            max = np.max(x_, axis=1)
            std = np.std(x_, axis=1)
            idx = sample_idx * num_segments + segment_idx
            dst_x[idx] = np.hstack((mean, max, std))
            dst_y[idx] = y

def cross_validate_feature_performance(train_set, test_set, num_col_size, normalize=False,
                                       populate_features=populate_features):
    num_train_samples = len(train_set)
    num_test_samples = len(test_set)
    num_segments = train_set[0][0].shape[0]
    
    # aggregate over all frames, calculate simple summary statistics
    # such as average, max or standard deviation for each channel
    x_train = np.empty((num_train_samples * num_segments, num_col_size * 3))
    y_train = np.empty((num_train_samples * num_segments,))
    x_test = np.empty((num_test_samples * num_segments, num_col_size * 3))
    y_test = np.empty((num_test_samples * num_segments,))

    populate_features(train_set, x_train, y_train, num_segments)
    populate_features(test_set, x_test, y_test, num_segments)

    cols_selections = sorted(tuple({
        'mean': np.arange(num_col_size),
        'max': np.arange(num_col_size, num_col_size * 2),
        'std': np.arange(num_col_size * 2, num_col_size * 3)
    }.items()))
    
    scores = {}
    for num_combs in range(1, 4):
        for cols in combinations(cols_selections, num_combs):
            name = '+'.join([col[0] for col in cols])
            print('Features:', name)
            cols = [col[1] for col in cols]
            cols = np.hstack(cols)
            x_train_ = np.take(x_train, cols, axis=1)
            x_test_ = np.take(x_test, cols, axis=1)

            if normalize:
                x_train_ = normalize(x_train_)
                x_test_ = normalize(x_test_)

            params = {
                'C': [10**x for x in range(-3, 3)],
                #'gamma': [10**x for x in range(-8, 0)],
                'random_state': [1234],
                'max_iter': [100],
            }

            rs = GridSearchCV(LinearSVC(), params, n_jobs=-1, cv=10, verbose=0)
            rs.fit(x_train_, y_train)

            svm = LinearSVC(**rs.best_params_)
            svm.fit(x_train_, y_train)
            
            train_score = accuracy_score(y_train, svm.predict(x_train_))
            test_score = accuracy_score(y_test, svm.predict(x_test_))

            print('Train score:', train_score)
            print('Test score:', test_score)
            scores[name] = {'train': train_score, 'test': test_score}
    return scores


def cross_validate_raw_performance(num_segments, random_seeds):
    print('Cross validate on RAW features.')
    scores = {}
    for random_seed in random_seeds:
        print('Using random seed:', random_seed)
        train_set = GTZAN_SPEC(phase='val', test_size=0.1, val_size=0.4,
                               min_segments=num_segments, randomized=True, random_seed=random_seed)
        test_set = GTZAN_SPEC(phase='test', test_size=0.1, val_size=0.4,
                              min_segments=num_segments, random_seed=random_seed)
        scores_ = cross_validate_feature_performance(train_set, test_set, 221)
        for name, phase_score in scores_.items():
            if name not in scores:
                scores[name] = {'train': [], 'test': []}
            for phase, score in phase_score.items():
                scores[name][phase].append(score)
    return scores


def cross_validate_mfcc_performance(num_segments, random_seeds):
    print('Cross validate on MFCC features.')
    scores = {}
    for random_seed in random_seeds:
        print('Using random seed:', random_seed)
        train_set = GTZAN_MFCC(phase='val', test_size=0.1, val_size=0.4,
                               min_segments=num_segments, randomized=True, random_seed=random_seed)
        test_set = GTZAN_MFCC(phase='test', test_size=0.1, val_size=0.4,
                              min_segments=num_segments, random_seed=random_seed)
        scores_ = cross_validate_feature_performance(train_set, test_set, 96)
        for name, phase_score in scores_.items():
            if name not in scores:
                scores[name] = {'train': [], 'test': []}
            for phase, score in phase_score.items():
                scores[name][phase].append(score)
    return scores


def cross_validate_cdbn_performance(num_segments, random_seeds):
    print('Cross validate on CDBN features.')
    scores = {}
    for random_seed in random_seeds:
        print('Using random seed:', random_seed)
        train_set = GTZAN_SPEC(phase='val', test_size=0.1, val_size=0.4,
                               min_segments=num_segments, randomized=True, random_seed=random_seed)
        test_set = GTZAN_SPEC(phase='test', test_size=0.1, val_size=0.4,
                              min_segments=num_segments, random_seed=random_seed)
        scores_ = cross_validate_feature_performance(train_set, test_set, 300,
                                                     populate_features=populate_cdbn_features)
        for name, phase_score in scores_.items():
            if name not in scores:
                scores[name] = {'train': [], 'test': []}
            for phase, score in phase_score.items():
                scores[name][phase].append(score)
    return scores


def merge_scores(scores, new_scores, feature_name):
    if feature_name not in scores:
        scores[feature_name] = {}
    scores = scores[feature_name]
    for name, phase_score in new_scores.items():
        if name not in scores:
            scores[name] = {'train': [], 'test': []}
        for phase, score in phase_score.items():
            scores[name][phase].extend(score)

            
def cross_validate(num_seeds=10):
    seeds = np.random.randint(0, int(1e5), num_seeds)
    print('Cross validate using seeds:', seeds)

    target_num_segments = [1, 2, 3, 5]
    scores = {x: {} for x in target_num_segments}
    for num_segments in target_num_segments:
        for feature_name, validate_func in (
                ('RAW', cross_validate_raw_performance),
                ('MFCC', cross_validate_mfcc_performance),
                ('CDBN', cross_validate_cdbn_performance),
                ):
            scores_ = validate_func(num_segments, seeds)
            merge_scores(scores[num_segments], scores_,
                         feature_name)
        with open('scores_segments_{}.pkl'.format(num_segments), 'wb') as f:
            pickle.dump(scores, f)
    with open('scores.pkl', 'wb') as f:
        pickle.dump(scores, f)
    return scores


if __name__ == '__main__':
    print(cross_validate())