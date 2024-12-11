from datetime import datetime

import numpy as np
import pandas as pd


def save_results_to_excel(result_file: str, output_dir: str = "evaluation_results"):
    """평가 결과를 엑셀 파일로 저장하는 함수"""

    # 결과를 저장할 데이터프레임 초기화
    recommendations_data = []
    metrics_data = []

    current_user = None
    test_items_count = None
    test_items_score = None

    # 결과 파일 읽기
    with open(result_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # 새로운 사용자 시작
        if line.startswith("User"):
            current_user = int(line.split()[1])

        # 테스트 아이템 평균 점수
        elif line.startswith("Average score for test items:"):
            test_items_score = float(line.split(": ")[1])

        # 테스트 아이템 개수
        elif line.startswith("Number of test items:"):
            test_items_count = int(line.split(": ")[1])

        # 추천 결과 파싱
        elif line and line[0].isdigit():
            try:
                parts = line.split()
                rank = int(parts[0])
                item_id = int(parts[1])
                score = float(parts[2])
                in_test = parts[3]

                recommendations_data.append(
                    {
                        "User ID": current_user,
                        "Test Items Count": test_items_count,
                        "Avg Test Items Score": test_items_score,
                        "Rank": rank,
                        "Item ID": item_id,
                        "Prediction Score": score,
                        "In Test Set": in_test,
                    }
                )
            except:
                continue

        # 전체 메트릭 파싱
        elif line.startswith(("MAP:", "Precision@", "Recall@")):
            metric_name = line.split(":")[0]
            metric_value = float(line.split(": ")[1])
            metrics_data.append({"Metric": metric_name, "Value": metric_value})

    # 데이터프레임 생성
    df_recommendations = pd.DataFrame(recommendations_data)
    df_metrics = pd.DataFrame(metrics_data)

    # 피벗 테이블 생성 (사용자별 추천 결과를 보기 쉽게 정리)
    df_pivot = df_recommendations.pivot_table(
        index=["User ID", "Test Items Count", "Avg Test Items Score"],
        columns="Rank",
        values=["Item ID", "Prediction Score"],
        aggfunc="first",
    ).round(4)

    # 현재 시간을 파일명에 포함
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f"{output_dir}/evaluation_results_{timestamp}.xlsx"

    # ExcelWriter로 여러 시트에 저장
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # 전체 추천 결과
        df_recommendations.to_excel(writer, sheet_name="Detailed Results", index=False)

        # 피벗 테이블 (사용자별 요약)
        df_pivot.to_excel(writer, sheet_name="User Summary")

        # 평가 지표
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)

        # 기본 통계 정보
        stats_data = {
            "Metric": [
                "Total Users",
                "Average Test Items per User",
                "Average Prediction Score",
                "Unique Recommended Items",
                "Most Common Recommended Item",
            ],
            "Value": [
                df_recommendations["User ID"].nunique(),
                df_recommendations.groupby("User ID")["Test Items Count"].first().mean(),
                df_recommendations["Prediction Score"].mean(),
                df_recommendations["Item ID"].nunique(),
                df_recommendations["Item ID"].mode().iloc[0],
            ],
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistics", index=False)

    print(f"Results saved to {excel_file}")
    return excel_file


if __name__ == "__main__":
    # 사용 예시
    result_file = "mlp_baseline/evaluation_results/evaluation_results_total.txt"
    save_results_to_excel(result_file)
