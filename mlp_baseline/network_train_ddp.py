import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from datasets import MLPDataset
from network import MLPRecommender
from scipy.io import mmread
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12353"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_model(
    rank,
    world_size,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
    patience=5,
):
    try:
        setup(rank, world_size)
        model = model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            ddp_model.train()
            train_loss = 0.0
            train_loader.sampler.set_epoch(epoch)

            train_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
                disable=rank != 0,
            )

            for user_ids, positive_item_ids, negative_item_ids in train_iterator:
                users = user_ids.to(rank)
                positive_item_ids = positive_item_ids.to(rank)
                negative_item_ids = negative_item_ids.to(rank)

                ratings = torch.cat(
                    [
                        torch.ones_like(positive_item_ids),
                        torch.zeros_like(negative_item_ids),
                    ],
                    dim=0,
                ).to(rank)
                items = torch.cat([positive_item_ids, negative_item_ids], dim=0).to(rank)

                scores = ddp_model(users, items)
                positive_scores = scores[: len(positive_item_ids)]
                negative_scores = scores[len(positive_item_ids) :]
                positive_loss = -torch.log(torch.sigmoid(positive_scores)).mean()
                negative_loss = -torch.log(1 - torch.sigmoid(negative_scores)).mean()
                loss = positive_loss + negative_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = train_loss / len(train_loader)

            if rank == 0:
                valid_loss = 0.0
                ddp_model.eval()

                with torch.no_grad():
                    valid_iterator = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
                    for batch in valid_iterator:
                        users = batch["user_id"].to(rank)
                        items = batch["item_id"].to(rank)
                        ratings = batch["label"].float().to(rank)

                        outputs = ddp_model(users, items)
                        loss = criterion(outputs, ratings.view(-1, 1))
                        valid_loss += loss.item()
                        valid_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                valid_loss = valid_loss / len(valid_loader)
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Valid Loss: {valid_loss:.4f}")

                scheduler.step(valid_loss)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": ddp_model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                        },
                        "best_network_model.pth",
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nEarly stopping after {epoch+1} epochs")
                        break

    finally:
        cleanup()
        torch.cuda.empty_cache()


def main(rank, world_size):
    # 데이터 로드
    train_matrix = mmread("/home/comoz/main_project/playlist_project/data2/split_data2/train_matrix.mtx").tocsr()
    valid_matrix = mmread("/home/comoz/main_project/playlist_project/data2/split_data2/valid_matrix.mtx").tocsr()

    num_users, num_items = train_matrix.shape
    if rank == 0:
        print(f"Users: {num_users}, Items: {num_items}")

    # 데이터셋 생성
    train_dataset = MLPDataset(train_matrix, num_negatives=4)
    valid_dataset = MLPDataset(valid_matrix, num_negatives=4)

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

    # DDP용 sampler 생성
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        if rank == 0
        else None
    )

    # 모델 초기화
    model = MLPRecommender(num_users=num_users, num_items=num_items)

    if rank == 0:
        print(model)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5, verbose=True)

    train_model(
        rank=rank,
        world_size=world_size,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        patience=10,
    )


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
