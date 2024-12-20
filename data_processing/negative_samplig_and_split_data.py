"""
Negative sampling and split data
"""

import pandas as pd
import numpy as np
import random

from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from tqdm import tqdm

def negative_sampling(data, num_epochs=5, num_negative_samples=4)->csr_matrix:
    """
    Negative sampling
    word2vec 방식 - 빈도수에 3/4승을 취해서 샘플링하여, 보다 빈도수가 적은 데이터도 샘플링되도록 함
    """

    num_users, num_items = data.shape

    item_frequencies = np.array(data.sum(axis=0)).flatten() + 1
    negative_frequencies = np.power(item_frequencies, 3/4)
    negative_frequencies = negative_frequencies / (negative_frequencies.sum() + 1e-10)

    all_items = set(range(num_items))
    
    row = []
    col = []
    data_values = []

    for user_idx in tqdm(range(num_users)):
        # 유저가 상호작용한 아이템 찾기
        user_items = set(data[user_idx, :].nonzero()[1])
        # 상호작용하지 않은 아이템들을 후보로
        candidate_items = list(all_items - user_items)

        if len(candidate_items) == 0:
            continue

        # 해당 유저의 positive 상호작용 수 계산
        num_positives = len(user_items)
        negative_sampling_num = num_positives * num_epochs * num_negative_samples

        # 후보 아이템들의 가중치 계산
        weights = negative_frequencies[candidate_items]
        weights = weights / (weights.sum() + 1e-10)

        # 네거티브 샘플링
        negative_items = np.random.choice(
            candidate_items,
            size=min(negative_sampling_num, len(candidate_items)),
            replace=False,
            p=weights
        )

        row.extend([user_idx] * len(negative_items))
        col.extend(negative_items)
        data_values.extend([-1] * len(negative_items))

    # 디버깅용 으로 첫번째 유저에 대한 아이템 출력
    print(f"User 0 items: {user_items}")
    print(f"First few negative samples: {negative_items[:5]}")

    negative_matrix = csr_matrix((data_values, (row, col)), shape=(num_users, num_items))
    return negative_matrix


def negative_sampling_and_split_data_valid(data, num_epochs=5, num_negative_samples=4)->tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """
    Negative sampling and split data

    """
    negative_data = negative_sampling(data, num_epochs, num_negative_samples)

    train_data, valid_data = split_data(data, ratio=0.875)
    negative_train, negative_valid = split_data(negative_data, ratio=0.875)

    return negative_train, negative_valid, train_data, valid_data

def split_data(data : csr_matrix, ratio=0.8)->tuple[csr_matrix, csr_matrix]:
    """
    Split data

    csr_matrix 데이터를 분할한다.
    """
    num_users = data.shape[0]
    train_indices = np.random.choice(num_users, size=int(num_users * ratio), replace=False)
    test_indices = np.setdiff1d(np.arange(num_users), train_indices)

    train_data = data[train_indices, :]
    test_data = data[test_indices, :]

    train_data = csr_matrix(train_data)
    test_data = csr_matrix(test_data)

    return train_data, test_data




if __name__ == "__main__":

    num_epochs = 5
    num_negative_samples = 4
    play_list_csr = mmread("/home/comoz/main_project/playlist_project/data/playlist_song_matrix_50_50.mtx").tocsr()
    train_data, test_data = split_data(play_list_csr)

    negative_train_data, negative_valid_data, train_data, valid_data = negative_sampling_and_split_data_valid(train_data, num_epochs, num_negative_samples)

    mmwrite("/home/comoz/main_project/playlist_project/data/split_data/negative_train_data.mtx", negative_train_data)
    mmwrite("/home/comoz/main_project/playlist_project/data/split_data/negative_valid_data.mtx", negative_valid_data)
    mmwrite("/home/comoz/main_project/playlist_project/data/split_data/train_data.mtx", train_data)
    mmwrite("/home/comoz/main_project/playlist_project/data/split_data/valid_data.mtx", valid_data)
    mmwrite("/home/comoz/main_project/playlist_project/data/split_data/test_data.mtx", test_data)