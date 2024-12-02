import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import MLPDataset
from models import MLPModel
from scipy.io import mmread



def calculate_metrics(model, test_matrix, train_matrix, valid_matrix, device, k_values=[1, 5, 10]):
    """모델 평가 함수"""
    model.eval()

    results = {
        "MAP": [],
        "Precision": {k: [] for k in k_values},
        "Recall": {k: [] for k in k_values},
    }

    num_users = test_matrix.shape[0]
    num_items = test_matrix.shape[1]
    batch_size = 256  # 배치 크기 설정

    # 전체 예측값을 저장할 텐서
    all_scores = torch.empty(num_users, num_items, device=device)

    # 배치 단위로 예측값 계산
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_users, batch_size), desc="Calculating scores"):
            end_idx = min(start_idx + batch_size, num_users)
            batch_users = torch.arange(start_idx, end_idx).to(device)
            
            # 배치의 모든 사용자-아이템 쌍 생성
            batch_users = (batch_users.unsqueeze(1).expand(-1, num_items) + 1).reshape(-1)  # 1-based
            batch_items = torch.arange(1, num_items + 1, device=device).repeat(end_idx - start_idx)  # 1-based
            
            # 예측값 계산
            batch_scores = model(batch_users, batch_items)
            all_scores[start_idx:end_idx] = batch_scores.reshape(-1, num_items)

    # 사용자별로 평가 수행
    for user_idx in tqdm(range(num_users), desc="Evaluating"):
        # 테스트 아이템 가져오기 (이미 1-based)
        test_items = set(test_matrix[user_idx].indices)
        if len(test_items) == 0:
            continue

        # 이미 본 아이템 가져오기 (이미 1-based)
        train_items = set(train_matrix[user_idx].indices)
        valid_items = set(valid_matrix[user_idx].indices)

        # 현재 사용자의 점수 복사
        scores = all_scores[user_idx].clone()

        # 학습/검증 아이템 마스킹
        scores[torch.tensor(list(train_items), device=device) - 1] = float("-inf")
        scores[torch.tensor(list(valid_items), device=device) - 1] = float("-inf")

        # 상위 K개 아이템 가져오기
        max_k = max(k_values)
        _, top_items = torch.topk(scores, k=max_k)
        recommended_items = (top_items.cpu().numpy() + 1).tolist()

        print(user_idx,recommended_items)

        # MAP 계산
        ap = 0.0
        hits = 0
        for rank, item in enumerate(recommended_items, 1):
            if item in test_items:
                hits += 1
                ap += hits / rank
        ap = ap / len(test_items)
        results["MAP"].append(ap)

        # Precision@K, Recall@K 계산
        for k in k_values:
            k_items = set(recommended_items[:k])
            num_hits = len(k_items & test_items)

            precision = num_hits / k
            recall = num_hits / len(test_items)

            results["Precision"][k].append(precision)
            results["Recall"][k].append(recall)

    # 최종 평균 계산
    final_results = {
        "MAP": np.mean(results["MAP"]),
        "Precision": {k: np.mean(v) for k, v in results["Precision"].items()},
        "Recall": {k: np.mean(v) for k, v in results["Recall"].items()},
    }

    return final_results


def main():
    """메인 평가 함수"""
    # 데이터 로드
    data_dir = "/home/comoz/main_project/playlist_project/data/split_data"
    train_matrix = mmread(f"{data_dir}/train_matrix.mtx").tocsr()
    valid_matrix = mmread(f"{data_dir}/valid_matrix.mtx").tocsr()
    test_matrix = mmread(f"{data_dir}/test_matrix.mtx").tocsr()

    # 모델 로드
    num_users, num_items = train_matrix.shape
    model = MLPModel(num_users + 1, num_items + 1)  # +1 for 1-based indexing

    # checkpoint에서 model_state_dict 추출
    checkpoint = torch.load("best_model_rank0.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 평가 실행
    metrics = calculate_metrics(model, test_matrix, train_matrix, valid_matrix, device)

    # 결과 출력
    print("\nEvaluation Results:")
    print(f"MAP: {metrics['MAP']:.4f}")
    for k in sorted(metrics["Precision"].keys()):
        print(f"Precision@{k}: {metrics['Precision'][k]:.4f}")
        print(f"Recall@{k}: {metrics['Recall'][k]:.4f}")


if __name__ == "__main__":
    main()
