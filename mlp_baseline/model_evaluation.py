from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import mmread
from scipy.sparse import csr_matrix
from tqdm import tqdm

from models import MLPModel

argparser = ArgumentParser("preprocess_dataset")
argparser.add_argument("--model-path", type=str, default="mlp_baseline/save_model/model2_test.pt")
argparser.add_argument("--device", type=str, default="cuda")
argparser.add_argument("--batch-size", type=int, default=1024)

def calculate_metrics(
    model: nn.Module,
    test_matrix: csr_matrix,
    train_matrix: csr_matrix,
    valid_matrix: csr_matrix,
    device: str,
    k_values: list[int] = [1, 5, 10],
    batch_size: int = 1024,
):
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

            # 이미 본 아이템 가져오기 (훈련 + 검증 데이터)
            train_items = set(train_matrix[user_idx].indices)
            valid_items = set(valid_matrix[user_idx].indices)
            seen_items = train_items.union(valid_items)

            # 평가에 사용할 아이템 리스트 생성
            candidate_items = np.setdiff1d(np.arange(num_items), list(seen_items))
            items_to_predict = np.array(list(test_items) + list(candidate_items))

            # 배치 처리
            all_scores = []

            for i in range(0, len(items_to_predict), batch_size):
                batch_items = items_to_predict[i : i + batch_size]

                # numpy array로 먼저 변환하여 속도 개선
                batch_items_array = np.array([batch_items])

                # 모델의 forward 함수 입력 형태에 맞게 수정
                user_tensor = torch.tensor([[user_idx]], dtype=torch.long, device=device)
                item_tensor = torch.from_numpy(batch_items_array).to(device=device, dtype=torch.long)

                # 예측값 계산 및 차원 처리
                batch_scores = model(user_tensor, item_tensor)
                # 항상 1차원 배열로 변환
                batch_scores = batch_scores.squeeze().cpu().numpy()
                if batch_scores.ndim == 0:  # 스칼라인 경우
                    batch_scores = np.array([batch_scores])
                all_scores.append(batch_scores)

            # 모든 배치의 점수를 합침
            scores = np.concatenate(all_scores)

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
    args = argparser.parse_args()
    # 데이터 로드
    data_dir = "/home/comoz/main_project/playlist_project/data/split_data"
    train_matrix = mmread(f"{data_dir}/train_matrix.mtx").tocsr()
    valid_matrix = mmread(f"{data_dir}/valid_matrix.mtx").tocsr()
    test_matrix = mmread(f"{data_dir}/test_matrix.mtx").tocsr()

    num_users, num_items = train_matrix.shape

    # save_model 디렉토리 내의 모든 .pt 파일 순회
    model_dir = Path("mlp_baseline/save_model")

    result_dir = Path("mlp_baseline/evaluation_results")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"evaluation_results.txt"

    # 평가 설정 정보 저장
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("-" * 50 + "\n\n")

    def _eval():
        model = MLPModel(num_users, num_items)

        # checkpoint에서 model_state_dict 추출
        checkpoint = torch.load(args.model_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # 평가 실행
        metrics = calculate_metrics(
            model,
            test_matrix,
            train_matrix,
            valid_matrix,
            device,
            batch_size=args.batch_size,
        )

        # 결과를 파일과 콘솔에 동시에 출력
        result_text = f"\nModel: {args.model_path}\n"
        result_text += f"MAP: {metrics['MAP']:.4f}\n"

        for k in sorted(metrics["Precision"].keys()):
            result_text += f"Precision@{k}: {metrics['Precision'][k]:.4f}\n"

        for k in sorted(metrics["Recall"].keys()):
            result_text += f"Recall@{k}: {metrics['Recall'][k]:.4f}\n"

        result_text += "-" * 50 + "\n"

        # 콘솔 출력
        print(result_text)

        # 파일 저장
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(result_text)
            f.flush()

    # 모든 모델 평가
    for model_path in sorted(model_dir.glob("*.pt")):
        print(f"\nEvaluating model: {model_path}")
        args.model_path = str(model_path)
        _eval()

    print(f"\nResults have been saved to: {result_file}")


if __name__ == "__main__":
    main()
