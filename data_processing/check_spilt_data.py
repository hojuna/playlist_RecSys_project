from scipy.io import mmread
import numpy as np

# 데이터 로드
train_matrix = mmread("data/split_data/train_data.mtx").tocsr()
valid_matrix = mmread("data/split_data/valid_data.mtx").tocsr()
test_matrix = mmread("data/split_data/test_data.mtx").tocsr()

# test 데이터의 각 요소 확인
test_rows, test_cols = test_matrix.nonzero()

for row, col in zip(test_rows, test_cols):
    # train과 valid에서 해당 요소 확인
    if train_matrix[row, col] != 0:
        print(f"요소 ({row}, {col})가 train 데이터에 존재합니다!")
    if valid_matrix[row, col] != 0:
        print(f"요소 ({row}, {col})가 valid 데이터에 존재합니다!")

# 중복 개수 계산
train_overlap = sum(1 for r, c in zip(test_rows, test_cols) if train_matrix[r, c] != 0)
valid_overlap = sum(1 for r, c in zip(test_rows, test_cols) if valid_matrix[r, c] != 0)

print(f"\n총 테스트 데이터 개수: {len(test_rows)}")
print(f"train과 중복: {train_overlap}")
print(f"valid와 중복: {valid_overlap}")
