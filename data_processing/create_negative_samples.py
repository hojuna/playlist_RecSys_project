import argparse
import os
import pickle
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix
from tqdm import tqdm

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--default-output-path", type=str, default="/home/comoz/main_project/playlist_project/data/negative_sample/default")
parser.add_argument("--raw-output-path", type=str, default="/home/comoz/main_project/playlist_project/data/negative_sample/raw")
args = parser.parse_args()
# fmt: on


def create_negative_samples(
    raw_matrix: csr_matrix,
    matrix: csr_matrix,
    num_epochs: int = 5,
    num_negative_samples: int = 4,
    sampling_range: str = "default",  # or raw
    matrix_name: str = "train",
    output_path: str = None,
):
    num_users, num_items = matrix.shape
    print(f"Matrix shape: Users={num_users}, Items={num_items}")

    # 상호작용이 없는 아이템의 빈도 계산
    item_frequencies = np.array(matrix.sum(axis=0)).flatten() + 1

    # 멱법칙 적용 및 정규화
    negative_frequencies = np.power(item_frequencies, 3 / 4)

    # 각 유저별 상호작용한 아이템 목록 생성
    user_interactions = {}
    for user_idx in range(num_users):
        # raw_matrix와 matrix에서 모든 상호작용 아이템 수집
        raw_items = set(raw_matrix[user_idx, :].nonzero()[1])
        matrix_items = set(matrix[user_idx, :].nonzero()[1])
        user_interactions[user_idx] = raw_items

    # 전체 아이템 ID 집합
    all_items = set(range(num_items))

    # 네거티브 샘플링을 위한 희소 행렬 초기화
    total_negative_samples = []

    for user_idx in tqdm(range(num_users)):
        # 유저가 상호작용하지 않은 아이템들
        user_items = user_interactions[user_idx]
        candidate_items = list(all_items - user_items)

        if len(candidate_items) == 0:
            continue

        # 해당 유저의 positive 상호작용 수 계산
        num_positives = len(matrix[user_idx, :].nonzero()[1])

        # negative sampling num per user
        negative_sampling_num = num_negative_samples * num_epochs * num_positives

        # 후보 아이템들의 가중치 계산
        weights = negative_frequencies[candidate_items]
        weights = weights / weights.sum()  # 후보 아이템들에 대해 다시 정규화

        num_sampled = 0
        negative_samples_per_user = []

        # 네거티브 샘플링
        while num_sampled < negative_sampling_num:
            negative_items = np.random.choice(
                candidate_items, size=min(negative_sampling_num, len(candidate_items)), replace=False, p=weights
            )

            for negative_item in negative_items:
                if negative_item not in user_items:
                    negative_samples_per_user.append(negative_item)
                    num_sampled += 1
                    if num_sampled == negative_sampling_num:
                        break

            # 첫 번째 유저에 대해서만 디버깅 출력
            if user_idx == 0:
                print(f"\nUser 0 debugging:")
                print(f"Number of positive interactions: {num_positives}")
                print(f"Number of negative samples: {len(negative_items)}")
                print(f"First few negative samples: {negative_items[:5]}")

        total_negative_samples.append(negative_samples_per_user)

    # CSR 행렬 생성
    with open(f"{output_path}/negative_samples.pkl", "wb") as f:
        pickle.dump(total_negative_samples, f)


def main():
    raw_matrix = mmread("/home/comoz/main_project/playlist_project/data/playlist_song_matrix_50_50.mtx").tocsr()
    train_matrix = mmread("/home/comoz/main_project/playlist_project/data/split_data/train_matrix.mtx").tocsr()
    valid_matrix = mmread("/home/comoz/main_project/playlist_project/data/split_data/valid_matrix.mtx").tocsr()

    num_epochs = 5
    num_negative_samples_list = [4, 8, 16]
    matrix_dict = {"train": train_matrix, "valid": valid_matrix}

    # for matrix_name, target_matrix in matrix_dict.items():
    #     for num_negative_samples in num_negative_samples_list:
    create_negative_samples(
        raw_matrix,
        train_matrix,
        num_epochs=num_epochs,
        num_negative_samples=4,
        matrix_name="train",
        output_path=args.default_output_path,
    )


if __name__ == "__main__":
    main()
