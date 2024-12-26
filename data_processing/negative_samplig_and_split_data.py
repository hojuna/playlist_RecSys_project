"""
Negative sampling and split data
"""

import pandas as pd
import numpy as np
import random

from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from tqdm import tqdm


def negative_sampling(data, num_epochs=5, num_negative_samples=4) -> csr_matrix:
    """
    Negative sampling
    word2vec 방식 - 빈도수에 3/4승을 취해서 샘플링하여, 보다 빈도수가 적은 데이터도 샘플링되도록 함
    """

    num_users, num_items = data.shape

    item_frequencies = np.array(data.sum(axis=0)).flatten() + 1
    negative_frequencies = np.power(item_frequencies, 3 / 4)
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
            candidate_items, size=min(negative_sampling_num, len(candidate_items)), replace=False, p=weights
        )

        row.extend([user_idx] * len(negative_items))
        col.extend(negative_items)
        data_values.extend([-1] * len(negative_items))

    # 디버깅용 으로 첫번째 유저에 대한 아이템 출력
    print(f"User 0 items: {user_items}")
    print(f"First few negative samples: {negative_items[:5]}")

    negative_matrix = csr_matrix((data_values, (row, col)), shape=(num_users, num_items))
    return negative_matrix


def negative_sampling_and_split_data_valid(
    data, num_epochs=5, num_negative_samples=4
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """
    Negative sampling and split data
    """

    # 먼저 train/test 분할 (0.8)
    train_data_raw, test_data = split_data(data, ratio=0.8)

    num_negative_samples_list = [4, 8, 16]

    for num_negative_samples_ in num_negative_samples_list:
        negative_data = negative_sampling(train_data_raw, num_epochs, num_negative_samples_)
        negative_train_final, negative_valid = split_data(negative_data, ratio=0.875)
        mmwrite(
            f"/home/comoz/main_project/playlist_project/data/split_data/negative_train_data_num_epochs_{num_epochs}_num_negative_samples_{num_negative_samples_}.mtx",
            negative_train_final,
        )
        mmwrite(
            f"/home/comoz/main_project/playlist_project/data/split_data/negative_valid_data_num_epochs_{num_epochs}_num_negative_samples_{num_negative_samples_}.mtx",
            negative_valid,
        )

    negative_data = negative_sampling(train_data_raw, num_epochs, num_negative_samples)

    # train 데이터를 다시 train/valid 분할 (0.875 = 0.7/0.8)
    train_final, valid_data = split_data(train_data_raw, ratio=0.875)

    # negative 데이터도 같은 비율로 분할

    return negative_train_final, negative_valid, train_final, valid_data, test_data


def split_data(data: csr_matrix, ratio=0.8) -> tuple[csr_matrix, csr_matrix]:
    """
    각 유저의 아이템 인터랙션을 비율에 따라 분할
    """
    train_indices = []
    test_indices = []
    train_data = []
    test_data = []
    rows = []
    test_rows = []

    # 각 유저별로 처리
    for user_idx in range(data.shape[0]):
        # 유저의 인터랙션 아이템 인덱스
        items = data[user_idx].indices
        n_items = len(items)

        if n_items == 0:  # 인터랙션이 없는 경우 스킵
            continue

        # 인터랙션을 랜덤하게 섞음
        perm = np.random.permutation(n_items)
        split_idx = int(n_items * ratio)

        # train set
        train_items = items[perm[:split_idx]]
        train_indices.extend(train_items)
        rows.extend([user_idx] * len(train_items))
        train_data.extend([1] * len(train_items))

        # test set
        test_items = items[perm[split_idx:]]
        test_indices.extend(test_items)
        test_rows.extend([user_idx] * len(test_items))
        test_data.extend([1] * len(test_items))

    # 한번에 희소 행렬 생성
    train_matrix = csr_matrix((train_data, (rows, train_indices)), shape=data.shape)

    test_matrix = csr_matrix((test_data, (test_rows, test_indices)), shape=data.shape)

    return train_matrix, test_matrix


if __name__ == "__main__":
    num_epochs = 10
    num_negative_samples = 4
    play_list_csr = mmread("/home/comoz/main_project/playlist_project/data/playlist_song_matrix_50_50.mtx").tocsr()

    negative_train, negative_valid, train_data, valid_data, test_data = negative_sampling_and_split_data_valid(
        play_list_csr, num_epochs, num_negative_samples
    )

    # 결과 저장
    # mmwrite("/home/comoz/main_project/playlist_project/data/split_data/negative_train_data.mtx", negative_train)
    # mmwrite("/home/comoz/main_project/playlist_project/data/split_data/negative_valid_data.mtx", negative_valid)
    # mmwrite("/home/comoz/main_project/playlist_project/data/split_data/train_data.mtx", train_data)
    # mmwrite("/home/comoz/main_project/playlist_project/data/split_data/valid_data.mtx", valid_data)
    # mmwrite("/home/comoz/main_project/playlist_project/data/split_data/test_data.mtx", test_data)

    # 분할 결과 출력
    print(f"Original data shape: {play_list_csr.shape}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"\nInteraction counts:")
    print(f"Original: {play_list_csr.sum()}")
    print(f"Train: {train_data.sum()}")
    print(f"Valid: {valid_data.sum()}")
    print(f"Test: {test_data.sum()}")
