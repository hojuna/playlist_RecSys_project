import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import MLPDataset
from models import MLPModel
from scipy.io import mmread
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_model(device, model, train_loader, valid_loader, optimizer, num_epochs, save_path, criterion):
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for user_ids, pos_items, neg_items, pos_mask, neg_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            user_ids = user_ids.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)

            optimizer.zero_grad()

            # Flatten tensors for model input
            users = user_ids.repeat_interleave(pos_mask.sum(dim=1) + neg_mask.sum(dim=1))
            items = torch.cat([
                pos_items[pos_mask],
                neg_items[neg_mask]
            ])

            # Labels
            labels = torch.cat([
                torch.ones(pos_mask.sum().item(), device=device),
                torch.zeros(neg_mask.sum().item(), device=device)
            ])

            # Forward pass
            outputs = model(users, items).squeeze()

            # Loss calculation
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        if valid_loader is not None:
            valid_loss = validate_model(model, valid_loader, device, criterion)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_valid_loss,
                    },
                    os.path.join(save_path, "best_model.pt"),
                )

            print(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
                f"Valid Loss = {valid_loss:.4f}"
            )


def validate_model(model, valid_loader, device, criterion):
    model.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for user_ids, pos_items, neg_items, pos_mask, neg_mask in valid_loader:
            user_ids = user_ids.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)

            # Flatten tensors for model input
            users = user_ids.repeat_interleave(pos_mask.sum(dim=1) + neg_mask.sum(dim=1))
            items = torch.cat([
                pos_items[pos_mask],
                neg_items[neg_mask]
            ])

            # Labels
            labels = torch.cat([
                torch.ones(pos_mask.sum().item(), device=device),
                torch.zeros(neg_mask.sum().item(), device=device)
            ])

            # Forward pass
            outputs = model(users, items).squeeze()

            # Loss calculation
            loss = criterion(outputs, labels)

            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_loader)
    return avg_valid_loss


def custom_collate_fn(batch):
    """
    서로 다른 길이의 positive/negative items를 처리하는 collate 함수
    """
    user_ids = []
    max_pos_len = max(len(x[1]) for x in batch)
    max_neg_len = max(len(x[2]) for x in batch)

    batch_size = len(batch)

    # User IDs 텐서
    user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)

    # Positive items 텐서 및 마스크 생성
    pos_items = torch.full((batch_size, max_pos_len), -1, dtype=torch.long)  # -1로 패딩
    pos_mask = torch.zeros((batch_size, max_pos_len), dtype=torch.bool)
    for i, (_, pos, _) in enumerate(batch):
        pos_items[i, :len(pos)] = torch.tensor(pos, dtype=torch.long)
        pos_mask[i, :len(pos)] = 1

    # Negative items 텐서 및 마스크 생성
    neg_items = torch.full((batch_size, max_neg_len), -1, dtype=torch.long)  # -1로 패딩
    neg_mask = torch.zeros((batch_size, max_neg_len), dtype=torch.bool)
    for i, (_, _, neg) in enumerate(batch):
        neg_items[i, :len(neg)] = torch.tensor(neg, dtype=torch.long)
        neg_mask[i, :len(neg)] = 1

    return user_ids, pos_items, neg_items, pos_mask, neg_mask

def main():
    data_dir = "/home/comoz/main_project/playlist_project/data/split_data"
    train_matrix = mmread(f"{data_dir}/train_matrix.mtx").tocsr()
    valid_matrix = mmread(f"{data_dir}/valid_matrix.mtx").tocsr()

    train_dataset = MLPDataset(train_matrix, num_negatives=2)
    valid_dataset = MLPDataset(valid_matrix, num_negatives=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    num_users, num_items = train_matrix.shape
    model = MLPModel(num_users + 1, num_items + 1, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-6,
    )

    train_model(device, model, train_loader, valid_loader, optimizer, num_epochs=10, save_path=".", criterion=criterion)

if __name__ == "__main__":
    main()
