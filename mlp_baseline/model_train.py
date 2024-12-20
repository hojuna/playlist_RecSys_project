import os
import random
from argparse import ArgumentParser, Namespace
from typing import Tuple
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import MLPDataset
from models import MLPModel
from scipy.io import mmread
from torch.utils.data import DataLoader
from tqdm import tqdm

# fmt: off
argparser = ArgumentParser("preprocess_dataset")
argparser.add_argument("--train-path", type=str, default="/home/comoz/main_project/playlist_project/data/split_data/train_data.mtx")
argparser.add_argument("--valid-path", type=str, default="/home/comoz/main_project/playlist_project/data/split_data/valid_data.mtx")
argparser.add_argument("--save-path", type=str, default="/home/comoz/main_project/playlist_project/mlp_baseline/save_model")
argparser.add_argument("--train-negative-path", type=str, default="/home/comoz/main_project/playlist_project/data/negative_sample/default/train_default_5_epochs_4_negatives.mtx")
argparser.add_argument("--valid-negative-path", type=str, default="/home/comoz/main_project/playlist_project/data/negative_sample/default/valid_default_5_epochs_4_negatives.mtx")
argparser.add_argument("--negative-mode", type=str, default="default")
argparser.add_argument("--num-epochs", type=int, default=5)
argparser.add_argument("--train-batch-size", type=int, default=1024)
argparser.add_argument("--valid-batch-size", type=int, default=4096)
argparser.add_argument("--num-negatives", type=int, default=4)  # 4, 8, 16
argparser.add_argument("--model-dim", type=int, default=128)
argparser.add_argument("--dropout", type=float, default=0.5)    
argparser.add_argument("--lr", type=float, default=1e-3)  # 1e-4, 1e-3, 5e-3, 1e-2, 1e-1
argparser.add_argument("--weight-decay", type=float, default=1e-4)
argparser.add_argument("--log-interval", type=int, default=50)
argparser.add_argument("--valid-interval", type=int, default=200)
argparser.add_argument("--num-workers", type=int, default=8)
argparser.add_argument("--prefetch-factor", type=int, default=4)
argparser.add_argument("--seed", type=int, default=42)
# fmt: on




def save_log(*args, **kwargs):
    result_file = f"mlp_baseline/training_results/training_log.txt"
    with open(result_file, "a", encoding='utf-8') as f:
        print(*args, **kwargs, file=f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_model(
    device: str,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    save_path: str,
    args: Namespace,
):
    model.train()

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_valid_loss = float("inf")

    step = 0
    avg_train_loss = torch.tensor(0.0, device=device)
    state=True
    for epoch in range(num_epochs):
        for user_ids, positive_ids, negative_ids in train_loader:
            user_ids = user_ids.to(device)
            positive_ids = positive_ids.to(device)
            negative_ids = negative_ids.to(device)
            if state:
                print(user_ids[:10],positive_ids[:10],negative_ids[:10])
                state=False

            total_ids = torch.cat([positive_ids, negative_ids], dim=1)
            # mixed precision training

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(user_ids, total_ids).squeeze()  # (batch_size, num_items)

                positive_labels = torch.ones_like(positive_ids, device=device, dtype=torch.float)
                negative_labels = torch.zeros_like(negative_ids, device=device, dtype=torch.float)
                positive_loss = criterion(outputs[:, : positive_ids.size(1)], positive_labels)
                negative_loss = criterion(outputs[:, positive_ids.size(1) :], negative_labels)
                loss = (positive_loss.mean(axis=1) + negative_loss.mean(axis=1)) / 2
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            avg_train_loss += loss.detach()

            if step % args.log_interval == 0:
                avg_train_loss = avg_train_loss / args.log_interval
                learning_rate = optimizer.param_groups[0]["lr"]
                print(f"Step {step}: Train Loss = {avg_train_loss:.4f}, Learning Rate = {learning_rate:.4f}")
                avg_train_loss = torch.tensor(0.0, device=device)

            if step % args.valid_interval == 0:

                # Validation phase
                model.eval()
                valid_loss, valid_positive_accuracy, valid_negative_accuracy = validate_model(
                    model, valid_loader, device
                )
                model.train()

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_valid_loss,
                        },
                        os.path.join(save_path, f"model_{args.negative_mode}_{args.num_negatives}_{args.lr}_lr.pt"),
                    )

                print(
                    f"Step {step}: Valid Loss = {valid_loss:.4f}, ",
                    f"Valid Positive Accuracy = {valid_positive_accuracy:.4f}, ",
                    f"Valid Negative Accuracy = {valid_negative_accuracy:.4f}",
                )

    # 모든 에포크 반복문이 끝난 후 최종 학습 결과 저장
    final_valid_loss, final_positive_accuracy, final_negative_accuracy = validate_model(
        model, valid_loader, device
    )

    # 최종 학습 결과를 로그 파일로 저장
    save_log(f"Final Training Results")
    save_log("-" * 50)
    save_log(f"Model Configuration:")
    save_log(f"Total Epochs: {args.num_epochs}")
    save_log(f"Learning Rate: {args.lr}")
    save_log(f"Number of Negatives: {args.num_negatives}")
    save_log(f"Negative Sampling Mode: {args.negative_mode}")
    save_log("-" * 50)
    save_log("")
    save_log("Final Validation Results:")
    save_log(f"Valid Loss: {final_valid_loss:.4f}")
    save_log(f"Valid Positive Accuracy: {final_positive_accuracy:.4f}")
    save_log(f"Valid Negative Accuracy: {final_negative_accuracy:.4f}")
    save_log(f"Best Valid Loss: {best_valid_loss:.4f}")
    save_log("-" * 50)

def validate_model(model: nn.Module, valid_loader: DataLoader, device: str) -> Tuple[float, float, float]:
    model.eval()

    avg_positive_accuracy = torch.tensor(0.0, device=device)
    avg_negative_accuracy = torch.tensor(0.0, device=device)
    total_valid_loss = torch.tensor(0.0, device=device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for user_ids, positive_ids, negative_ids in valid_loader:
            user_ids = user_ids.to(device)
            positive_ids = positive_ids.to(device)
            negative_ids = negative_ids.to(device)

            positive_labels = torch.ones_like(positive_ids, device=device, dtype=torch.float)
            negative_labels = torch.zeros_like(negative_ids, device=device, dtype=torch.float)

            total_ids = torch.cat([positive_ids, negative_ids], dim=1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(user_ids, total_ids).squeeze()

            pos_outputs = outputs[:, : positive_ids.size(1)]
            neg_outputs = outputs[:, positive_ids.size(1) :]

            pos_loss = criterion(pos_outputs, positive_labels).mean(axis=1)
            neg_loss = criterion(neg_outputs, negative_labels).mean(axis=1)

            pos_accuracy = (pos_outputs > 0).float().sum() / positive_ids.size(1)
            neg_accuracy = (neg_outputs < 0).float().sum() / negative_ids.size(1)

            avg_positive_accuracy += pos_accuracy
            avg_negative_accuracy += neg_accuracy

            loss = (pos_loss + neg_loss) / 2
            total_valid_loss += loss.sum()
    avg_valid_loss = total_valid_loss.item() / len(valid_loader.dataset)
    avg_positive_accuracy = avg_positive_accuracy.item() / len(valid_loader.dataset)
    avg_negative_accuracy = avg_negative_accuracy.item() / len(valid_loader.dataset)

    return avg_valid_loss, avg_positive_accuracy, avg_negative_accuracy


def main():

    result_file = f"mlp_baseline/training_results/training_log.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)


    args = argparser.parse_args()
    torch.cuda.empty_cache()

    set_seed(args.seed)

    # 데이터 로드
    train_matrix = mmread(args.train_path).tocsr()
    valid_matrix = mmread(args.valid_path).tocsr()



    def _train():
        # with open("/home/comoz/main_project/playlist_project/data/negative_sample/default/negative_samples.pkl", "rb") as f:
        #     train_negative_matrix = pickle.load(f)

        # with open(args.valid_negative_path, "rb") as f:
        #     valid_negative_matrix = pickle.load(f)

        train_negative_matrix = mmread("/home/comoz/main_project/playlist_project/data/split_data/negative_train_data.mtx").tocsr()
        valid_negative_matrix = mmread("/home/comoz/main_project/playlist_project/data/split_data/negative_valid_data.mtx").tocsr()

        # 데이터셋 생성
        train_dataset = MLPDataset(train_matrix, train_negative_matrix, num_negatives=args.num_negatives)
        valid_dataset = MLPDataset(valid_matrix, valid_negative_matrix, num_negatives=args.num_negatives)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")

        # 데이터로더 설정
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
        )

        # 모델 설정
        num_users, num_items = train_matrix.shape
        model = MLPModel(num_users, num_items, model_dim=args.model_dim, dropout=args.dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))

        # 모델 학습
        train_model(
            device=device,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            save_path=args.save_path,
            args=args,
        )
    lr_list = [1e-4, 1e-3, 5e-3, 1e-2, 1e-1]
    num_negatives_list = [4, 8, 16]
    negative_mode_list = ["default"]
    _train()

    # for lr in lr_list:
    #     for num_negatives in num_negatives_list:
    #         for negative_mode in negative_mode_list:
    #             args.lr = lr
    #             args.num_negatives = num_negatives
    #             args.negative_mode = negative_mode
    #             args.train_negative_path = f"/home/comoz/main_project/playlist_project/data/negative_sample/{args.negative_mode}/train_{args.negative_mode}_5_epochs_{args.num_negatives}_negatives.mtx"
    #             args.valid_negative_path = f"/home/comoz/main_project/playlist_project/data/negative_sample/{args.negative_mode}/valid_{args.negative_mode}_5_epochs_{args.num_negatives}_negatives.mtx"

    #             _train()


if __name__ == "__main__":

    main()
