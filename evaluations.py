import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
import scipy.sparse as sp
import pickle
from scipy.io import mmread
from scipy.linalg import solve



def load_model_and_data(model_path='/home/comoz/main_project/playlist_project/model/nmf_model.pkl', 
                       test_path='/home/comoz/main_project/playlist_project/data/dataset/split_data/test_matrix.mtx',
                       train_path='/home/comoz/main_project/playlist_project/data/dataset/split_data/train_matrix.mtx',
                       user_factors_path='/home/comoz/main_project/playlist_project/model/user_factors.npy',
                       item_factors_path='/home/comoz/main_project/playlist_project/model/item_factors.npy'
                       ):
    """
    저장된 모델과 테스트 데이터를 불러옵니다.
    """
    # 모델 로드
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        
    user_factors = np.load(user_factors_path)
    item_factors = np.load(item_factors_path)
    
    # 테스트 데이터 로드
    test_matrix = mmread(test_path).tocsr()
    train_matrix = mmread(train_path).tocsr()
    
    return user_factors, item_factors, test_matrix, train_matrix


def calculate_metrics(H, test_matrix, train_matrix, k_list=[1, 5, 10], lambda_reg=0.1):
    """
    정규방정식으로 W를 구한 후 K-precision, K-recall, MAP 계산
    train에 사용된 아이템은 제외
    """
    # 전체 사용자의 W 한번에 계산
    HHt = H @ H.T + lambda_reg * np.eye(H.shape[0])
    W_test = test_matrix @ H.T @ np.linalg.inv(HHt)
    
    # 예측값 계산
    predicted_matrix = W_test @ H
    
    n_users = predicted_matrix.shape[0]
    results = {f'precision@{k}': [] for k in k_list}
    results.update({f'recall@{k}': [] for k in k_list})
    results['MAP'] = []

    for user_idx in tqdm(range(n_users)):
        # train에서 사용된 아이템 인덱스
        train_items = train_matrix[user_idx].nonzero()[1]
        
        # 테스트 세트에서 실제 청취한 곡들
        actual_items = test_matrix[user_idx].nonzero()[1]
        
        if len(actual_items) == 0:
            continue
            
        # 현재 사용자의 예측 점수
        pred_scores = predicted_matrix[user_idx].copy()
        
        # train에서 사용된 아이템은 -inf로 마스킹
        pred_scores[train_items] = -np.inf
        
        # train 아이템을 제외한 실제 테스트 아이템
        actual_items = np.setdiff1d(actual_items, train_items)
        if len(actual_items) == 0:  # train에 없는 새로운 아이템이 없으면 건너뛰기
            continue
        
        # 예측 점수로 정렬하여 상위 K개 아이템 인덱스 추출
        recommended_items = np.argsort(pred_scores)[::-1]
        
        # k별로 precision, recall 계산
        for k in k_list:
            top_k_items = recommended_items[:k]
            hits = np.isin(top_k_items, actual_items)
            
            precision_k = np.sum(hits) / k
            results[f'precision@{k}'].append(precision_k)
            
            recall_k = np.sum(hits) / len(actual_items)
            results[f'recall@{k}'].append(recall_k)
        
        # MAP 계산
        ap = 0
        hits = 0
        for i, item in enumerate(recommended_items, 1):
            if item in actual_items:
                hits += 1
                ap += hits / i
        
        if len(actual_items) > 0:
            ap /= len(actual_items)
        results['MAP'].append(ap)
    
    # 평균값 계산
    final_results = {metric: np.mean(values) for metric, values in results.items() if values}
    
    return final_results

def print_metrics(metrics):
    """평가 지표 출력"""
    for metric, value in sorted(metrics.items()):
        print(f"{metric}: {value:.4f}")  # 소수점 4자리까지만 출력

if __name__ == "__main__":
    # 모델과 데이터 로드
    print("모델과 데이터를 불러오는 중...")
    _, item_factors, test_matrix, train_matrix = load_model_and_data()
    
    # 평가 지표 계산
    print("평가 지표를 계산하는 중...")
    metrics = calculate_metrics(
        item_factors, 
        test_matrix, 
        train_matrix, 
        k_list=[1, 5, 10]
    )
    print_metrics(metrics)