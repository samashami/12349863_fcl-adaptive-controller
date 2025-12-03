import argparse
import csv
import os
from datetime import datetime
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from data import (
    DataConfig,
    load_cifar100,
    split_train_val,
    make_iid_client_splits,
    build_dataloaders_for_client,
    build_global_test_loader,
)
from model import build_model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    return total_correct / total_samples if total_samples > 0 else 0.0


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits, y)
        total_loss += loss.item()
        total_acc += batch_acc
        total_batches += 1

    return {
        "loss": total_loss / max(1, total_batches),
        "acc": total_acc / max(1, total_batches),
    }


def create_run_dir(base_dir: str = "results/runs") -> str:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def init_csv_logger(run_dir: str) -> str:
    csv_path = os.path.join(run_dir, "baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cl_batch",
                "epoch",
                "train_loss",
                "train_acc",
                "val_acc",
                "test_acc",
            ]
        )
    return csv_path


def append_csv_row(csv_path: str, row: List[Any]) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline FCL CIFAR-100 training")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-cl-batches", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs-per-batch", type=int, default=2)
    parser.add_argument("--client-id", type=int, default=0, help="Which client to simulate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Data config and loading ---
    cfg = DataConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_clients=args.num_clients,
        num_cl_batches=args.num_cl_batches,
    )

    train_full, test = load_cifar100(cfg)
    train, val = split_train_val(train_full, cfg.val_size)

    client_splits = make_iid_client_splits(train, cfg.num_clients)
    client_indices = client_splits[args.client_id]

    cl_train_loaders, val_loader = build_dataloaders_for_client(
        train, val, client_indices, cfg
    )
    test_loader = build_global_test_loader(test, cfg)

    # --- Model and optimizer ---
    model = build_model(num_classes=100, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Logging setup ---
    run_dir = create_run_dir()
    csv_path = init_csv_logger(run_dir)

    print(f"Run directory: {run_dir}")

    # --- Continual training over CL batches for one client ---
    for cl_batch_idx, train_loader in enumerate(cl_train_loaders):
        print(f"\n=== CL batch {cl_batch_idx + 1}/{len(cl_train_loaders)} ===")

        for epoch in range(args.epochs_per_batch):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            test_acc = evaluate(model, test_loader, device)

            print(
                f"[CL {cl_batch_idx} | epoch {epoch}] "
                f"loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} "
                f"val_acc={val_acc:.4f} "
                f"test_acc={test_acc:.4f}"
            )

            append_csv_row(
                csv_path,
                [
                    cl_batch_idx,
                    epoch,
                    train_metrics["loss"],
                    train_metrics["acc"],
                    val_acc,
                    test_acc,
                ],
            )

    print("\nTraining finished.")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()