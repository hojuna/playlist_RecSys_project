import os
import pickle
from argparse import ArgumentParser, Namespace

import numpy as np
from scipy.io import mmread, mmwrite
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

argparser = ArgumentParser("train_nmf_model")
argparser.add_argument("--file-path", type=str, default="data/split_data")
argparser.add_argument("--save-path", type=str, default="nmf/nmf_models")
argparser.add_argument("--n-components", type=int, default=64)
argparser.add_argument("--max-iter", type=int, default=500)
argparser.add_argument("--train-matrix", type=str, default="train_matrix.mtx")
argparser.add_argument("--valid-matrix", type=str, default="valid_matrix.mtx")


def validate_model(H, valid_matrix, train_matrix, lambda_reg=0.1):
    """
    검증 데이터로 모델 성능 평가
    """
    # 전체 사용자의 W 한번에 계산
    HHt = H @ H.T + lambda_reg * np.eye(H.shape[0])
    W_valid = valid_matrix @ H.T @ np.linalg.inv(HHt)

    # 예측값 계산
    pred_matrix = W_valid @ H

    # train에서 사용된 위치는 마스킹l
    train_mask = train_matrix.nonzero()
    pred_matrix[train_mask] = 0
    valid_matrix_array = valid_matrix.toarray()
    valid_matrix_array[train_mask] = 0

    # 실제 값이 있고 train에 있는 위치만 선택해서 RMSE 계산
    mask = valid_matrix_array > 0
    rmse = np.sqrt(np.mean((pred_matrix[mask] - valid_matrix_array[mask]) ** 2))

    return rmse


def train_nmf_model(args: Namespace) -> tuple[NMF, np.ndarray, np.ndarray]:

    # 데이터 로드
    train_matrix = mmread(os.path.join(args.file_path, args.train_matrix)).tocsr()
    valid_matrix = mmread(os.path.join(args.file_path, args.valid_matrix)).tocsr()

    # NMF 모델 초기화
    nmf_model = NMF(
        n_components=args.n_components,  # 잠재 요인의 수
        max_iter=args.max_iter,  # 최대 반복 횟수
        random_state=42,  # 재현성을 위한 시드
    )

    # 모델 학습
    # W: 사용자-잠재요인 행렬, H: 잠재요인-아이템 행렬
    W = nmf_model.fit_transform(train_matrix)
    H = nmf_model.components_

    valid_rmse = validate_model(H, valid_matrix, train_matrix)
    print(f"검증 데이터 RMSE: {valid_rmse:.4f}")

    return nmf_model, W, H


def model_save(model: NMF, user_factors: np.ndarray, item_factors: np.ndarray, args: Namespace) -> None:
    np.save(os.path.join(args.save_path, "user_factors.npy"), user_factors)
    np.save(os.path.join(args.save_path, "item_factors.npy"), item_factors)
    with open(os.path.join(args.save_path, "nmf_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print("user_factors, item_factors, 모델 저장 완료!")


if __name__ == "__main__":
    args = argparser.parse_args()
    model, user_factors, item_factors = train_nmf_model(args)
    model_save(model, user_factors, item_factors, args)
