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
        for user_idx in range(self.num_users):
            for item_idx in interaction_matrix[user_idx].indices:
                self.data.append((user_idx, item_idx))
                self.items_per_user[user_idx].append(item_idx)
        self.num_data = len(self.data)

        self.negative_matrix = negative_matrix

    def __len__(self):
        return self.num_data

    def _get_negative_items(self, user_idx: int, num_samples: int) -> np.ndarray:
        """저장된 네거티브 매트릭스에서 가져오기"""
        negative_items = self.negative_matrix[user_idx].indices[:num_samples]
        return negative_items

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_id (int): 유저 ID
            pos_items (np.ndarray): 해당 유저의 포지티브 아이템 리스트
            neg_items (np.ndarray): 샘플링된 네거티브 아이템 리스트
        """
        user_id, positive_ids = self.data[index]

        # 포지티브 아이템 개수 * num_negatives 만큼 네거티브 샘플링
        negative_ids = self._get_negative_items(user_id, self.num_negatives)

        user_id = torch.tensor([user_id], dtype=torch.long)
        positive_ids = torch.tensor([positive_ids], dtype=torch.long)
        negative_ids = torch.tensor(negative_ids, dtype=torch.long)

        return user_id, positive_ids, negative_ids
