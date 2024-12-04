import numpy as np
import torch
from tqdm import tqdm
from models import MLPModel
from scipy.io import mmread

def calculate_metrics(model, test_matrix, train_matrix, valid_matrix, device, k_values=[1, 5, 10], num_negative=100):
    """모델 평가 함수"""
    model.eval()

    results = {
        "MAP": [],
        "Precision": {k: [] for k in k_values},
        "Recall": {k: [] for k in k_values},
    }

    num_users = test_matrix.shape[0]
    num_items = test_matrix.shape[1]

    with torch.no_grad():
        for user_idx in tqdm(range(num_users), desc="Evaluating"):
            # 테스트 아이템 가져오기
            test_items = set(test_matrix[user_idx].indices)
            if len(test_items) == 0:
                continue

            # 이미 본 아이템 가져오기
            train_items = set(train_matrix[user_idx].indices)
            valid_items = set(valid_matrix[user_idx].indices)
            
            seen_items = train_items.union(valid_items)

            mask_items= np.array([416,1092,368])

            seen_items = train_items.union(mask_items)

            # # 평가에 사용할 아이템 리스트 생성 (negative sampling)
            negative_items = np.setdiff1d(np.arange(num_items), list(seen_items.union(test_items)))
            
            sampled_negative_items = np.random.choice(negative_items, size=min(len(negative_items), num_negative), replace=False)
            # items_to_predict = np.array(list(test_items) + list(sampled_negative_items)

            items_to_predict = np.array(list(test_items) + list(negative_items))

             # 평가에 사용할 모든 아이템 리스트 생성 (전체 네거티브)

            # negative_items = np.setdiff1d(np.arange(num_items), list(seen_items))
            # items_to_predict = np.array(list(test_items) + list(negative_items))
     
            # 사용자와 아이템 텐서 생성
            user_tensor = torch.full((len(items_to_predict),), user_idx, dtype=torch.long, device=device)
            item_tensor = torch.tensor(items_to_predict, dtype=torch.long, device=device)

            # 예측값 계산
            scores = model(user_tensor, item_tensor).squeeze().cpu().numpy()

            # 아이템과 점수를 정렬
            item_score_dict = dict(zip(items_to_predict, scores))
            ranked_items = sorted(item_score_dict, key=item_score_dict.get, reverse=True)

            # 평가 지표 계산
            ap = 0.0
            hits = 0
            for rank, item in enumerate(ranked_items, 1):
                if item in test_items:
                    hits += 1
                    ap += hits / rank
            ap = ap / len(test_items)
            results["MAP"].append(ap)

            for k in k_values:
                top_k_items = ranked_items[:k]
                print(top_k_items)
                num_hits = len(set(top_k_items) & test_items)
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
    model = MLPModel(num_users, num_items, dropout=0.3, embedding_dim=128)

    # checkpoint에서 model_state_dict 추출
    checkpoint = torch.load("best_model2.pt")
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

    for k in sorted(metrics["Recall"].keys()):
        print(f"Recall@{k}: {metrics['Recall'][k]:.4f}")

if __name__ == "__main__":
    main()
