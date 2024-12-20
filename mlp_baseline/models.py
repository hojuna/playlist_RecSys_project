import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, num_users, num_items, model_dim=128, dropout=0.5):
        super(MLPModel, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, model_dim)
        self.item_embedding = nn.Embedding(num_items, model_dim)

        self.user_embedding.weight.data.uniform_(-0.005, 0.005)
        self.item_embedding.weight.data.uniform_(-0.005, 0.005)

        self.user_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.item_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.final = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1),
        )

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # user_indices: (batch_size, 1) -> (batch_size, num_items)
        user_indices = user_indices.repeat(1, item_indices.size(1))

        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)

        x_user = self.user_mlp(user_embeds)
        x_item = self.item_mlp(item_embeds)

        x = torch.cat([x_user, x_item], dim=2)

        x = self.final(x)
        return x
