import random
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(
        self,
        interaction_matrix: csr_matrix,
        negative_matrix: csr_matrix,
        num_negatives: int = 4,
    ):
        """
        MLP 추천 모델을 위한 데이터셋

        Args:
            interaction_matrix: 유저-아이템 상호작용 행렬 (scipy sparse matrix, CSR format)
            num_negatives: 각 포지티브 샘플당 네거티브 샘플 수
        """
        # 행렬 정보 저장
        self.num_users, self.num_items = interaction_matrix.shape
        self.num_negatives = num_negatives

        self.song_ids = set(range(self.num_items))

        # 유저별 포지티브 아이템 리스트 생성
        self.data = []
        self.items_per_user = defaultdict(list)
        self.neg_items_per_user = defaultdict(list)

        # for user_idx in range(self.num_users):
        #     for item_idx in interaction_matrix[user_idx].indices:
        #         self.data.append((user_idx, item_idx))
        #         self.items_per_user[user_idx].append(item_idx)
        #     neg_items = negative_matrix[user_idx]
        #     self.neg_items_per_user[user_idx].extend(np.random.choice(neg_items, size=self.num_negatives, replace=False))

        for user_idx in range(self.num_users):
            # 포지티브 아이템 처리
            for item_idx in interaction_matrix[user_idx].indices:
                self.data.append((user_idx, item_idx))
                self.items_per_user[user_idx].append(item_idx)

            # 네거티브 아이템 처리
            for neg_item_idx in negative_matrix[user_idx].indices:
                self.neg_items_per_user[user_idx].append(neg_item_idx)

        self.num_data = len(self.data)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_id (int): 유저 ID
            pos_items (np.ndarray): 해당 유저의 포지티브 아이템 리스트
            neg_items (np.ndarray): 샘플링된 네거티브 아이템 리스트
        """
        user_id, positive_ids = self.data[index]

        # num_negatives 개수만큼만 랜덤 선택
        negative_ids = np.random.choice(
            self.neg_items_per_user[user_id], size=self.num_negatives, replace=False  # 필요한 경우 중복 허용
        )

        user_id = torch.tensor([user_id], dtype=torch.long)
        positive_ids = torch.tensor([positive_ids], dtype=torch.long)
        negative_ids = torch.tensor(negative_ids, dtype=torch.long)

        return user_id, positive_ids, negative_ids
