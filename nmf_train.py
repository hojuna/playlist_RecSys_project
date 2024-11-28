from scipy.io import mmread, mmwrite
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle


def validate_model(H, valid_matrix, train_matrix, lambda_reg=0.1):
    """
    검증 데이터로 모델 성능 평가
    """
    # 전체 사용자의 W 한번에 계산
    HHt = H @ H.T + lambda_reg * np.eye(H.shape[0])
    W_valid = valid_matrix @ H.T @ np.linalg.inv(HHt)
    
    # 예측값 계산
    pred_matrix = W_valid @ H
    
    # train에서 사용된 위치는 마스킹
    train_mask = train_matrix.nonzero()
    pred_matrix[train_mask] = 0
    valid_matrix_array = valid_matrix.toarray()
    valid_matrix_array[train_mask] = 0
    
    # 실제 값이 있고 train에 없는 위치만 선택해서 RMSE 계산
    mask = (valid_matrix_array > 0)
    rmse = np.sqrt(np.mean((pred_matrix[mask] - valid_matrix_array[mask]) ** 2))
    
    return rmse


def train_nmf_model(
    config: dict
):
    # 데이터 로드
    train_matrix = mmread('/home/comoz/main_project/playlist_project/data/dataset/split_data/train_matrix.mtx').tocsr()
    valid_matrix = mmread('/home/comoz/main_project/playlist_project/data/dataset/split_data/valid_matrix.mtx').tocsr()
    
    # NMF 모델 초기화
    nmf_model = NMF(
        n_components=config['n_components'],  # 잠재 요인의 수
        max_iter=config['max_iter'],     # 최대 반복 횟수
        random_state=42  # 재현성을 위한 시드
        # init=config['init'],
        # alpha_W=config['alpha_W'],     # W 행렬에 대한 L2 정규화 파라미터
        # alpha_H=config['alpha_H']      # H 행렬에 대한 L2 정규화 파라미터
    )
    
    # 모델 학습
    # W: 사용자-잠재요인 행렬, H: 잠재요인-아이템 행렬
    W = nmf_model.fit_transform(train_matrix)
    H = nmf_model.components_
    
    valid_rmse = validate_model(H, valid_matrix, train_matrix)
    print(f"검증 데이터 RMSE: {valid_rmse:.4f}")
    
    return nmf_model, W, H

def model_save(model, user_factors, item_factors):
    np.save('/home/comoz/main_project/playlist_project/model/user_factors.npy', user_factors)
    np.save('/home/comoz/main_project/playlist_project/model/item_factors.npy', item_factors)
    with open('/home/comoz/main_project/playlist_project/model/nmf_model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    config = {  
        'n_components': 50,
        'max_iter': 500,
        # 'init': 'random',
        # 'alpha_W': 0.01,
        # 'alpha_H': 0.01
    }
    model, user_factors, item_factors = train_nmf_model(config)
    model_save(model, user_factors, item_factors)