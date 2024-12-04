import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, dropout=0.1):
        super(MLPModel, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)

        self.mlp=nn.Sequential(
            nn.Linear(embedding_dim , embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.final = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 8, embedding_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 8, 1),
        )

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)

        x_user=self.mlp(user_embeds)
        x_item=self.mlp(item_embeds)
        x = torch.cat([x_user, x_item], dim=1)
        x = self.final(x)
        return x
