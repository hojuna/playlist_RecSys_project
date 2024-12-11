import argparse
import os

import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--default-output-path", type=str, default="/home/comoz/main_project/playlist_project/data/negative_sample/default"
)
parser.add_argument(
    "--raw-output-path", type=str, default="/home/comoz/main_project/playlist_project/data/negative_sample/raw"
)
args = parser.parse_args()


def create_negative_samples(
    raw_matrix: csr_matrix,
    matrix: csr_matrix,
    num_epochs: int = 5,
    num_negative_samples: int = 4,
    sampling_range: str = "default",  # or raw
    matrix_name: str = "train",
    output_path: str = None,
):

    num_items, num_users = raw_matrix.shape

    # 상호작용이 없는 아이템의 빈도 계산
    item_frequencies = np.array(matrix.sum(axis=1)).flatten()
    negative_frequencies = np.ones(num_items) - (item_frequencies > 0).astype(float)

    # 빈든 값이 0인 경우 처리
    if negative_frequencies.sum() == 0:
        negative_frequencies = np.ones(num_items)  # 모든 아이템에 동일한 가중치 부여

    # 빈도가 0인 아이템들의 빈도를 1로 설정
    negative_frequencies = np.power(negative_frequencies, 3 / 4)  # 멱법칙 적용
    negative_frequencies = negative_frequencies / (negative_frequencies.sum() + 1e-10)  # 정규화 (epsilon 추가)

    # 샤플링 범위 설정
    if sampling_range == "default":
        target_matrix = matrix
    elif sampling_range == "raw":
        target_matrix = raw_matrix

    # 각 유저별 상호작용한 아이템 목록 생성
    user_interactions = {}
    for user_idx in range(num_users):
        user_items = set(target_matrix[:, user_idx].nonzero()[0])
        user_interactions[user_idx] = user_items

    # 전체 아이템 ID 집합
    all_items = set(range(num_items))

    # 네거티브 샘플링을 위한 희소 행렬 초기화
    rows = []
    cols = []
    data = []
    for user_idx in range(num_users):

        # 유저가 상호작용하지 않은 아이템들
        user_items = user_interactions[user_idx]
        candidate_items = list(all_items - user_items)

        # negative sampling num per user
        negative_sampling_num = num_negative_samples * num_epochs * len(matrix[:, user_idx].nonzero()[0])

        # 네거티브 아이템들의 가중치 계산
        weights = negative_frequencies[candidate_items]
        weights = weights / weights.sum()  # 후보 아이템들에 대해 다시 정규화

        negative_items = np.random.choice(
            candidate_items, size=min(negative_sampling_num, len(candidate_items)), replace=False, p=weights
        )

        rows.extend([user_idx] * len(negative_items))
        cols.extend(negative_items)
        data.extend([-1] * len(negative_items))

    # CSR 행렬 생성
    negative_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    filename = f"{matrix_name}_{sampling_range}_{num_epochs}_epochs_{num_negative_samples}_negatives.mtx"
    mmwrite(os.path.join(output_path, filename), negative_matrix)

    print(f"Negative samples saved to {filename}")


def main():

    raw_matrix = mmread("/home/comoz/main_project/playlist_project/data/playlist_song_matrix_50_50.mtx").tocsr()

    train_matrix = mmread("/home/comoz/main_project/playlist_project/data/split_data/train_matrix.mtx").tocsr()
    valid_matrix = mmread("/home/comoz/main_project/playlist_project/data/split_data/valid_matrix.mtx").tocsr()

    num_epochs = 5
    num_negative_samples_list = [4, 8, 16]
    matrix_dict = {"train": train_matrix, "valid": valid_matrix}

    for matrix_name, target_matrix in matrix_dict.items():
        for num_negative_samples in num_negative_samples_list:
            create_negative_samples(
                raw_matrix,
                target_matrix,
                num_epochs=num_epochs,
                num_negative_samples=num_negative_samples,
                matrix_name=matrix_name,
                output_path=args.default_output_path,
            )

    for matrix_name, target_matrix in matrix_dict.items():
        for num_negative_samples in num_negative_samples_list:
            create_negative_samples(
                raw_matrix,
                target_matrix,
                num_epochs=num_epochs,
                num_negative_samples=num_negative_samples,
                matrix_name=matrix_name,
                sampling_range="raw",
                output_path=args.raw_output_path,
            )


if __name__ == "__main__":
    main()
