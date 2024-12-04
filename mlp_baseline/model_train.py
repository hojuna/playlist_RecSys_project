import os
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import MLPDataset
from models import MLPModel
from scipy.io import mmread
from torch.utils.data import DataLoader
from tqdm import tqdm

argparser = ArgumentParser("preprocess_dataset")
argparser.add_argument("--train-path", type=str, default="data/split_data/train_matrix.mtx")
argparser.add_argument("--valid-path", type=str, default="data/split_data/valid_matrix.mtx")
argparser.add_argument("--save-path", type=str, default="mlp_baseline/save_model")
argparser.add_argument("--num-epochs", type=int, default=10)
argparser.add_argument("--batch-size", type=int, default=128)
argparser.add_argument("--num-negatives", type=int, default=4)
argparser.add_argument("--embedding-dim", type=int, default=32)
argparser.add_argument("--dropout", type=float, default=0.5)
argparser.add_argument("--lr", type=float, default=0.001)
argparser.add_argument("--device", type=str, default="cuda")
argparser.add_argument("--alpha", type=float, default=1.0)


def train_model(
    device: str,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    save_path: str,
    alpha: float,
):
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for user_ids, pos_items, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            user_ids = user_ids.to(device)
            optimizer.zero_grad()

            # Positive items 처리
            pos_losses = []
            for i, pos_item_list in enumerate(pos_items):
                if len(pos_item_list) == 0:
                    continue
                user_tensor = user_ids[i].repeat(len(pos_item_list))
                pos_item_tensor = torch.tensor(pos_item_list).long().to(device)
                pos_outputs = model(user_tensor, pos_item_tensor).squeeze()
                pos_losses.append(-torch.log(F.sigmoid(pos_outputs)).mean())
                print(f"pos_outputs: {pos_outputs}")

            # Negative items 처리
            neg_losses = []
            for i, neg_item_list in enumerate(neg_items):
                if len(neg_item_list) == 0:
                    continue
                user_tensor = user_ids[i].repeat(len(neg_item_list))
                neg_item_tensor = torch.tensor(neg_item_list).long().to(device)
                neg_outputs = model(user_tensor, neg_item_tensor).squeeze()
                neg_losses.append(-torch.log(1 - F.sigmoid(neg_outputs)).mean())

            # Total loss
            pos_loss = torch.stack(pos_losses).mean() if pos_losses else 0
            neg_loss = torch.stack(neg_losses).mean() if neg_losses else 0
            loss = pos_loss + neg_loss * alpha
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        if valid_loader is not None:
            valid_loss = validate_model(model, valid_loader, device, alpha)

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

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Valid Loss = {valid_loss:.4f}")


def validate_model(model: nn.Module, valid_loader: DataLoader, device: str, alpha: float) -> float:
    model.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for user_ids, pos_items, neg_items in valid_loader:
            user_ids = user_ids.to(device)

            # Positive items 처리
            pos_losses = []
            for i, pos_item_list in enumerate(pos_items):
                if len(pos_item_list) == 0:
                    continue
                user_tensor = user_ids[i].repeat(len(pos_item_list))
                pos_item_tensor = torch.tensor(pos_item_list).long().to(device)
                pos_outputs = model(user_tensor, pos_item_tensor).squeeze()
                pos_losses.append(-torch.log(F.sigmoid(pos_outputs)).mean())

            # Negative items 처리
            neg_losses = []
            for i, neg_item_list in enumerate(neg_items):
                if len(neg_item_list) == 0:
                    continue
                user_tensor = user_ids[i].repeat(len(neg_item_list))
                neg_item_tensor = torch.tensor(neg_item_list).long().to(device)
                neg_outputs = model(user_tensor, neg_item_tensor).squeeze()
                neg_losses.append(-torch.log(1 - F.sigmoid(neg_outputs)).mean())

            # Total loss
            pos_loss = torch.stack(pos_losses).mean() if pos_losses else 0
            neg_loss = torch.stack(neg_losses).mean() if neg_losses else 0
            loss = pos_loss + neg_loss * alpha
            total_valid_loss += loss.item()

    avg_valid_loss = total_valid_loss / len(valid_loader)
    return avg_valid_loss


def custom_collate_fn(
    batch: List[Tuple[int, List[int], List[int]]]
) -> Tuple[torch.Tensor, List[List[int]], List[List[int]]]:
    # 배치에서 각 컴포넌트를 분리
    user_ids, pos_items_lists, neg_items_lists = zip(*batch)

    # user_ids만 텐서로 변환하고 나머지는 리스트 형태로 유지
    user_ids = torch.tensor(list(user_ids)).long()

    return user_ids, list(pos_items_lists), list(neg_items_lists)


def main():
    args = argparser.parse_args()
    torch.cuda.empty_cache()

    # 데이터 로드
    train_matrix = mmread(args.train_path).tocsr()
    valid_matrix = mmread(args.valid_path).tocsr()

    # 데이터셋 생성
    train_dataset = MLPDataset(train_matrix, num_negatives=args.num_negatives)
    valid_dataset = MLPDataset(valid_matrix, num_negatives=args.num_negatives)

    # 데이터로더 설정
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    # 모델 설정
    num_users, num_items = train_matrix.shape
    model = MLPModel(num_users, num_items, embedding_dim=args.embedding_dim, dropout=args.dropout)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # 모델 학습
    train_model(
        device=device,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
