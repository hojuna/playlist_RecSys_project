import torch
import torch.nn as nn


class MLPRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, dropout=0.3):
        super(MLPRecommender, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp_user = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mlp_item = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.final = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_indices, item_indices):
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        user_x = self.mlp_user(user_emb)
        item_x = self.mlp_item(item_emb)

        # Concatenate embeddings
        x = torch.cat([user_x, item_x], dim=-1)

        # Final prediction
        prediction = self.final(x)

        return prediction.squeeze()
