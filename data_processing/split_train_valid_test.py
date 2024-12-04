import os
from argparse import ArgumentParser, Namespace

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split

argparser = ArgumentParser("split_data_by_user")
argparser.add_argument("--file-path", type=str, default="data/playlist_song_matrix_50_50.mtx")
argparser.add_argument("--save-path", type=str, default="data/split_data")


def split_data_by_user(sparse_matrix, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """사용자별로 아이템을 train/valid/test로 분할"""

    n_users = sparse_matrix.shape[0]
    train_matrix = []
    valid_matrix = []
    test_matrix = []

    for user_idx in range(n_users):
        # 현재 사용자의 아이템 인덱스 가져오기
        user_items = sparse_matrix[user_idx].nonzero()[1]
        n_items = len(user_items)

        if n_items < 3:  # 아이템이 3개 미만인 경우 모두 학습 데이터로
            n_train = n_items
            n_valid = 0
            n_test = 0
        else:
            # 비율에 따라 개수 계산
            n_train = int(n_items * train_ratio)
            n_valid = int(n_items * valid_ratio)
            n_test = n_items - n_train - n_valid

        # 아이템 인덱스를 랜덤하게 섞기
        np.random.shuffle(user_items)

        # 분할된 인덱스로 희소 행렬 생성
        train_data = csr_matrix(
            (np.ones(n_train), (np.zeros(n_train), user_items[:n_train])),
            shape=(1, sparse_matrix.shape[1]),
        )

        valid_data = csr_matrix(
            (
                np.ones(n_valid),
                (np.zeros(n_valid), user_items[n_train : n_train + n_valid]),
            ),
            shape=(1, sparse_matrix.shape[1]),
        )

        test_data = csr_matrix(
            (np.ones(n_test), (np.zeros(n_test), user_items[n_train + n_valid :])),
            shape=(1, sparse_matrix.shape[1]),
        )

        train_matrix.append(train_data)
        valid_matrix.append(valid_data)
        test_matrix.append(test_data)

    # 모든 사용자의 데이터를 하나의 행렬로 결합
    train_matrix = vstack(train_matrix)
    valid_matrix = vstack(valid_matrix)
    test_matrix = vstack(test_matrix)

    return train_matrix, valid_matrix, test_matrix


def print_split_result(
    sparse_matrix: csr_matrix, train_matrix: csr_matrix, valid_matrix: csr_matrix, test_matrix: csr_matrix
) -> None:

    # 결과 출력
    print("원본 데이터 shape:", sparse_matrix.shape)
    print("훈련 데이터 shape:", train_matrix.shape)
    print("검증 데이터 shape:", valid_matrix.shape)
    print("테스트 데이터 shape:", test_matrix.shape)

    # 각 행렬의 비제로 요소 수 확인
    print("\n각 행렬의 아이템 수:")
    print("원본:", sparse_matrix.nnz)
    print("훈련:", train_matrix.nnz)
    print("검증:", valid_matrix.nnz)
    print("테스트:", test_matrix.nnz)


def save_data(
    train_matrix: csr_matrix,
    valid_matrix: csr_matrix,
    test_matrix: csr_matrix,
    save_path: str,
) -> None:
    # 데이터 저장
    mmwrite(os.path.join(save_path, "train_matrix.mtx"), train_matrix)
    mmwrite(os.path.join(save_path, "valid_matrix.mtx"), valid_matrix)
    mmwrite(os.path.join(save_path, "test_matrix.mtx"), test_matrix)


# 메인 코드
if __name__ == "__main__":

    args = argparser.parse_args()

    # MTX 형식의 희소 행렬 파일 불러오기
    sparse_matrix = mmread(args.file_path).tocsr()

    # 데이터 분할
    train_matrix, valid_matrix, test_matrix = split_data_by_user(sparse_matrix)

    # 분할 결과 출력
    print_split_result(sparse_matrix, train_matrix, valid_matrix, test_matrix)

    # 원본 데이터와 분할된 데이터를 파일로 저장
    save_data(train_matrix, valid_matrix, test_matrix, args.save_path)

    print("데이터 분할 완료!")
