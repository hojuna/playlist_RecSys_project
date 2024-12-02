import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from datasets import MLPDataset
from models import MLPModel
from scipy.io import mmread
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np


def setup(rank, world_size):
    """프로세스 그룹 초기화"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """프로세스 그룹 정리"""
    dist.destroy_process_group()


def train_model(rank, world_size, ddp_model, train_loader, valid_loader, optimizer, num_epochs, save_path, criterion):
    """모델 학습 함수"""
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")
        ddp_model.train()
        total_loss = 0
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (users, pos_items_list, neg_items_list) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch+1}", disable=False
        )):
                
            optimizer.zero_grad()

            batch_loss = 0
            num_users_in_batch = len(users)
            for user, pos_items, neg_items in zip(users, pos_items_list, neg_items_list):
                # Move tensors to device
                user_tensor = torch.tensor([user] * (len(pos_items) + len(neg_items)), dtype=torch.long).to(rank)
                # 이미 텐서인 경우 clone().detach() 사용
                pos_items = pos_items.clone().detach().to(rank)
                neg_items = neg_items.clone().detach().to(rank)

                # Prepare inputs
                all_items = torch.cat([pos_items, neg_items])

                # Get predictions
                scores = ddp_model(user_tensor, all_items)

                # Calculate loss
                num_pos = len(pos_items)
                positive_scores = scores[:num_pos]
                negative_scores = scores[num_pos:]

                positive_loss = criterion(positive_scores, torch.ones_like(positive_scores))
                negative_loss = criterion(negative_scores, torch.zeros_like(negative_scores))
                loss = positive_loss + (negative_loss)

                # positive_loss = -torch.log(positive_scores).mean()
                # negative_loss = -torch.log(1 - negative_scores).mean()
                # loss = positive_loss + negative_loss

                batch_loss += loss

            # Backward pass
            batch_loss = batch_loss / num_users_in_batch
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()


        # Validation phase (only on rank 0)
        if rank == 0 and valid_loader is not None:
            valid_loss = validate_model(ddp_model, valid_loader, rank, criterion)
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ddp_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_valid_loss,
                    },
                    f"{save_path}/best_model_rank{rank}.pt",
                )

            print(
                f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, "
                f"Valid Loss = {valid_loss:.4f}"
            )


def validate_model(model, valid_loader, device, criterion):
    """검증 함수 분리"""
    model.eval()
    total_valid_loss = 0
    num_valid_batches = 0

    with torch.no_grad():
        for users, pos_items_list, neg_items_list in valid_loader:
            batch_loss = 0
            num_users_in_batch = len(users)
            for user, pos_items, neg_items in zip(users, pos_items_list, neg_items_list):
                # Move tensors to device
                user_tensor = torch.tensor([user] * (len(pos_items) + len(neg_items)), dtype=torch.long).to(device)
                # Use clone().detach() like in train_model
                pos_items = pos_items.clone().detach().to(device)
                neg_items = neg_items.clone().detach().to(device)

                # Prepare inputs
                all_items = torch.cat([pos_items, neg_items])
                scores = model(user_tensor, all_items)

                num_pos = len(pos_items)
                positive_scores = scores[:num_pos]
                negative_scores = scores[num_pos:]

                # positive_loss = -torch.log(positive_scores).mean()
                # negative_loss = -torch.log(1 - negative_scores).mean()
                # batch_loss += positive_loss + negative_loss
                positive_loss = criterion(positive_scores, torch.ones_like(positive_scores))
                negative_loss = criterion(negative_scores, torch.zeros_like(negative_scores))
                batch_loss = positive_loss + (negative_loss)

            # Backward pass
            batch_loss = batch_loss / num_users_in_batch
            total_valid_loss += batch_loss
            num_valid_batches += 1

    return total_valid_loss / num_valid_batches


def custom_collate_fn(batch):
    """서로 다른 길이의 데이터를 처리하는 collate 함수"""
    users, pos_items, neg_items = zip(*batch)
    return (
        torch.tensor(users),
        [torch.tensor(pos) for pos in pos_items],
        [torch.tensor(neg) for neg in neg_items]
    )


def main(rank, world_size):
    """메인 학습 함수"""
    # 프로세스 그룹 초기화
    setup(rank, world_size)

    # 데이터 로드 및 데이터셋 생성
    data_dir = "/home/comoz/main_project/playlist_project/data/split_data"
    train_matrix = mmread(f"{data_dir}/train_matrix.mtx").tocsr()
    valid_matrix = mmread(f"{data_dir}/valid_matrix.mtx").tocsr()

    # 데이터셋과 데이터 로더 생성
    train_dataset = MLPDataset(train_matrix, num_negatives=2)
    valid_dataset = MLPDataset(valid_matrix, num_negatives=2)

    # DataLoader 설정
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=custom_collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    # 모델 설정
    num_users, num_items = train_matrix.shape
    model = MLPModel(num_users + 1, num_items + 1, dropout=0.3)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=0.001,
        weight_decay=1e-6,
    )

    train_model(rank, world_size, ddp_model, train_loader, valid_loader, optimizer, num_epochs=10, save_path=".", criterion=criterion)
    cleanup()


if __name__ == "__main__":
    world_size = 1  # 단일 GPU 사용
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
