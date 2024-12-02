import numpy as np
import torch
from network import MLPRecommender
from scipy.io import mmread
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def calculate_metrics(model, test_matrix, train_matrix, device, k_values=[1, 5, 10], batch_size=4096):
    model.eval()
    results = {
        "MAP": [],
        "Precision": {k: [] for k in k_values},
        "Recall": {k: [] for k in k_values},
    }

    num_users, num_items = test_matrix.shape

    with torch.no_grad():
        for user_idx in tqdm(range(num_users), desc="Evaluating"):
            test_items = set(test_matrix[user_idx].indices)
            if not test_items:  # 테스트 아이템이 없으면 스킵
                continue

            train_items = set(train_matrix[user_idx].indices)

            # train 아이템을 제외한 실제 테스트 아이템
            test_items = test_items - train_items
            if not test_items:  # train에 없는 새로운 아이템이 없으면 건너뛰기
                continue

            if len(test_items) < 10:  # 10개 미만인 경우 스킵
                continue

            scores = torch.zeros(num_items, device=device)
            user_tensor = torch.tensor([user_idx] * batch_size, device=device)

            for i in range(0, num_items, batch_size):
                batch_size_i = min(batch_size, num_items - i)
                if batch_size_i != batch_size:
                    user_tensor = torch.tensor([user_idx] * batch_size_i, device=device)
                item_tensor = torch.tensor(range(i, i + batch_size_i), device=device)

                batch_scores = model(user_tensor, item_tensor)
                scores[i : i + batch_size_i] = batch_scores.squeeze()

            scores[list(train_items)] = float("-inf")

            recommended_items = torch.argsort(scores, descending=True).cpu().numpy()

            ap = 0
            hits = 0
            for i, item in enumerate(recommended_items, 1):
                if item in test_items:
                    hits += 1
                    ap += hits / i

            ap /= len(test_items)
            results["MAP"].append(ap)

            num_test_items = len(test_items)
            for k in sorted(k_values):
                pred_items = recommended_items[:k]
                num_hits = len(set(pred_items) & test_items)

                precision = num_hits / k
                recall = num_hits / num_test_items

                results["Precision"][k].append(precision)
                results["Recall"][k].append(recall)

    final_metrics = {"MAP": np.mean(results["MAP"]) if results["MAP"] else 0}

    for k in k_values:
        final_metrics[f"Precision@{k}"] = np.mean(results["Precision"][k]) if results["Precision"][k] else 0
        final_metrics[f"Recall@{k}"] = np.mean(results["Recall"][k]) if results["Recall"][k] else 0

    return final_metrics


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    interaction_matrix = mmread("/home/comoz/main_project/playlist_project/data2/split_data2/test_matrix.mtx").tocsr()
    num_users, num_items = interaction_matrix.shape
    print(f"Users: {num_users}, Items: {num_items}")

    model = MLPRecommender(num_users=num_users, num_items=num_items).to(device)

    checkpoint = torch.load("best_network_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = calculate_metrics(
        model=model,
        test_matrix=interaction_matrix,
        train_matrix=mmread("/home/comoz/main_project/playlist_project/data2/split_data2/train_matrix.mtx").tocsr(),
        device=device,
        k_values=[1, 5, 10],
    )

    print("\n=== Evaluation Results ===")
    print(f"\nMean Average Precision: {metrics['MAP']:.4f}")

    print("\nPrecision Metrics:")
    for k in [1, 5, 10]:
        print(f"Precision@{k}: {metrics[f'Precision@{k}']:.4f}")

    print("\nRecall Metrics:")
    for k in [1, 5, 10]:
        print(f"Recall@{k}: {metrics[f'Recall@{k}']:.4f}")


if __name__ == "__main__":
    main()
