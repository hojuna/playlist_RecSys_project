import os
import pickle
from argparse import ArgumentParser, Namespace

import numpy as np
from scipy.io import mmread
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from tqdm import tqdm

argparser = ArgumentParser("nmf_evaluations")
argparser.add_argument("--model-path", type=str, default="nmf/nmf_models/nmf_model.pkl")
argparser.add_argument("--data-path", type=str, default="data/split_data")


def load_model_and_data(args: Namespace) -> tuple[NMF, csr_matrix, csr_matrix, csr_matrix]:
    """모델과 데이터 로드"""

    data_path = args.data_path
    model_path = args.model_path

    # 테스트 데이터 로드
    test_matrix = mmread(os.path.join(data_path, "test_matrix.mtx")).tocsr()
    train_matrix = mmread(os.path.join(data_path, "train_matrix.mtx")).tocsr()
    valid_matrix = mmread(os.path.join(data_path, "valid_matrix.mtx")).tocsr()

    # 모델 로드
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, test_matrix, train_matrix, valid_matrix


def calculate_metrics(H, test_matrix, train_matrix, valid_matrix, k_list=[1, 5, 10], lambda_reg=0.1):
    """
    NMF 모델 성능 평가
    Args:
        H: 학습된 아이템 임베딩 행렬 (n_components x n_items)
        test_matrix: 테스트 데이터 행렬 (n_users x n_items)
        train_matrix: 훈련 데이터 행렬 (n_users x n_items)
        valid_matrix: 검증 데이터 행렬 (n_users x n_items)
        k_list: Precision과 Recall을 계산할 k값 리스트
        lambda_reg: 정규화 계수
    Returns:
        dict: 평가 지표 결과
    """
    # 전체 사용자의 W 한번에 계산
    HHt = H @ H.T + lambda_reg * np.eye(H.shape[0])
    W_test = test_matrix @ H.T @ np.linalg.inv(HHt)

    # 예측값 계산
    predicted_matrix = W_test @ H

    n_users = predicted_matrix.shape[0]
    results = {f"precision@{k}": [] for k in k_list}
    results.update({f"recall@{k}": [] for k in k_list})
    results["MAP"] = []

    for user_idx in tqdm(range(n_users)):
        # train에서 사용된 아이템 인덱스와 테스트 세트에서 실제 청취한 곡들
        train_items = set(train_matrix[user_idx].indices)
        valid_items = set(valid_matrix[user_idx].indices)
        actual_items = set(test_matrix[user_idx].indices)

        if not actual_items:
            continue

        # 현재 사용자의 예측 점수
        pred_scores = predicted_matrix[user_idx].copy()

        # train과 validation에서 사용된 아이템은 -inf로 마스킹
        pred_scores[list(train_items)] = -np.inf
        pred_scores[list(valid_items)] = -np.inf

        # train 아이템을 제외한 실제 테스트 아이템
        actual_items = actual_items - train_items - valid_items
        if not actual_items:  # train에 없는 새로운 아이템이 없으면 건너뛰기
            continue

        # 전체 추천 아이템 리스트 (내림차순 정렬)
        recommended_items = np.argsort(-pred_scores)

        # MAP 계산
        ap = 0
        hits = 0
        for i, item in enumerate(recommended_items, 1):
            if item in actual_items:
                hits += 1
                ap += hits / i

        if len(actual_items) > 0:
            ap /= len(actual_items)
        results["MAP"].append(ap)

        # k값별 Precision, Recall 계산
        actual_items_len = len(actual_items)  # 길이 캐싱
        for k in k_list:
            top_k_items = recommended_items[:k]
            hits = sum(1 for item in top_k_items if item in actual_items)

            # Precision@k, Recall@k
            results[f"precision@{k}"].append(hits / k)
            results[f"recall@{k}"].append(hits / actual_items_len)

    # 평균 계산
    return {metric: np.mean(values) for metric, values in results.items()}


def print_metrics(metrics):
    """평가 지표 출력"""
    print("\n=== 평가 결과 ===")

    if "MAP" in metrics:
        print(f"MAP: {metrics['MAP']:.4f}")

    print("\nPrecision:")
    for k in [1, 5, 10]:
        if f"precision@{k}" in metrics:
            print(f"  @{k}: {metrics[f'precision@{k}']:.4f}")

    print("\nRecall:")
    for k in [1, 5, 10]:
        if f"recall@{k}" in metrics:
            print(f"  @{k}: {metrics[f'recall@{k}']:.4f}")


if __name__ == "__main__":

    args = argparser.parse_args()
    # 모델과 데이터 로드
    print("모델과 데이터를 불러오는 중...")
    model, test_matrix, train_matrix, valid_matrix = load_model_and_data(args)

    # item factors 가져오기
    item_factors = model.components_

    # 평가 지표 계산
    print("평가 지표를 계산하는 중...")
    metrics = calculate_metrics(item_factors, test_matrix, train_matrix, valid_matrix)

    # 결과 출력
    print_metrics(metrics)
