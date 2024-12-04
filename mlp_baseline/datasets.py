from typing import Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(self, interaction_matrix: csr_matrix, num_negatives: int = 4):
        """
        MLP 추천 모델을 위한 데이터셋

        Args:
            interaction_matrix: 유저-아이템 상호작용 행렬 (scipy sparse matrix, CSR format)
            num_negatives: 각 포지티브 샘플당 네거티브 샘플 수
        """
        # 행렬 정보 저장
        self.num_users, self.num_items = interaction_matrix.shape
        self.num_negatives = num_negatives
        self.interaction_matrix = interaction_matrix

        # 유저별 포지티브 아이템 리스트 생성
        self.user_positive_items = {}
        for user_idx in range(self.num_users):
            positive_items = interaction_matrix[user_idx].indices
            if len(positive_items) > 0:  # 포지티브 아이템이 있는 유저만 포함
                self.user_positive_items[user_idx] = positive_items

        # 학습에 사용할 유저 리스트 (포지티브 아이템이 있는 유저만)
        self.users = list(self.user_positive_items.keys())

    def __len__(self):
        return len(self.users)

    def _get_negative_items(self, user_idx: int, num_samples: int) -> np.ndarray:
        """특정 유저에 대한 여러 개의 네거티브 아이템을 샘플링"""
        user_items = set(self.interaction_matrix[user_idx].indices)
        negative_items = set()

        while len(negative_items) < num_samples:
            item_idx = np.random.randint(0, self.num_items)
            if item_idx not in user_items:
                negative_items.add(item_idx)

        return np.array(list(negative_items))

    def __getitem__(self, index):
        """
        Returns:
            user_id (int): 유저 ID
            pos_items (np.ndarray): 해당 유저의 포지티브 아이템 리스트
            neg_items (np.ndarray): 샘플링된 네거티브 아이템 리스트
        """
        user_id = self.users[index]
        pos_items = self.user_positive_items[user_id]

        # 포지티브 아이템 개수 * num_negatives 만큼 네거티브 샘플링
        neg_items = self._get_negative_items(user_id, len(pos_items) * self.num_negatives)

        return user_id, pos_items, neg_items
