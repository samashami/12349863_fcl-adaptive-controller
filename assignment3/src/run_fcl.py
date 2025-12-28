# src/run_llm_fcl.py
import argparse, numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import optim
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
import random
import csv, os
import pandas as pd
import time

from assignment3.src.model import build_resnet18
from assignment3.src.fl import Client, Server
from assignment3.src.strategies.replay import ReplayBuffer
# from assignment3.src.policy import Policy  # not used in Assignment 3 baseline demo

# top of file (after imports)
GLOBAL_SEED = 42
def seed_worker(worker_id: int):
    import numpy as _np, random as _random
    _np.random.seed(GLOBAL_SEED + worker_id)
    _random.seed(GLOBAL_SEED + worker_id)

def evaluate(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    n_classes = 100
    hits = np.zeros(n_classes, dtype=np.int64)
    counts = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for c in range(n_classes):
                mask = (y == c)
                if mask.any():
                    counts[c] += mask.sum().item()
                    hits[c] += (pred[mask] == c).sum().item()
    acc = correct / total
    per_class_recall = np.array([ (hits[c]/counts[c]) if counts[c] > 0 else 0.0 for c in range(n_classes) ], dtype=np.float32)
    return acc, per_class_recall

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic behavior (safe even on CPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset_per_client", type=int, default=4000, help="for fast CPU demo; use -1 for all")
    ap.add_argument("--use_policy", action="store_true", help="enable LLM-like policy controller")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=200, help="batches between progress prints")
    ap.add_argument("--split_mode", choices=["equal", "dirichlet"], default="equal",
                help="equal: equal-size random split per client; dirichlet: non-iid alpha")
    ap.add_argument("--val_size", type=int, default=5000, help="validation holdout from CIFAR100 train")
    ap.add_argument("--cl_batches", type=int, default=7, help="number of continual-learning batches per client")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (paper: 4)")
    ap.add_argument("--optimizer", choices=["adam","sgd"], default="adam",
                help="Paper fine-tuning used Adam; switch to sgd if you want FedSGD baseline.")
    ap.add_argument("--early_patience", type=int, default=5)
    ap.add_argument("--tag", type=str, default="baseline",
                help="label for this run (used in CSV filenames)")
    

    args = ap.parse_args()
    set_seeds(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # make seed visible to worker processes
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed

    # seeded DataLoader workers
    g = torch.Generator()
    g.manual_seed(args.seed)

    # def _seed_worker(worker_id: int):
    #     np.random.seed(args.seed + worker_id)
    #     random.seed(args.seed + worker_id)


    # transforms
    tf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),  # ImageNet stats
    ])
    tf_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    
    # ===== datasets & paper-faithful split =====
    trainset_full = datasets.CIFAR100(root="./data", train=True, download=True, transform=tf_train)
    testset       = datasets.CIFAR100(root="./data", train=False, download=True, transform=tf_test)

    # --- hold out validation from CIFAR-100 train ---
    total_train = len(trainset_full)  # 50_000
    val_size = args.val_size          # 5_000 (paper)
    train_size = total_train - val_size

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_subset, val_subset = torch.utils.data.random_split(
        trainset_full, [train_size, val_size], generator=g
    )

    # absolute indices into trainset_full
    train_indices = np.array(train_subset.indices, dtype=np.int64)
    val_indices   = np.array(val_subset.indices, dtype=np.int64)

    # loaders for val/test (val not yet used; will be in early stopping step)
    valset = Subset(trainset_full, val_indices)
    val_loader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"[Split] train={len(train_indices)} val={len(val_indices)} test={len(testset)}", flush=True)

    # --- client splits (equal-size, random) ---
    if args.split_mode == "equal":
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(train_indices)
        sizes = [len(perm) // args.clients] * args.clients
        for i in range(len(perm) % args.clients):
            sizes[i] += 1
        splits = []
        start = 0
        for s in sizes:
            splits.append(perm[start:start+s].tolist())
            start += s
    else:
        raise NotImplementedError("dirichlet over 45k subset comes later; use --split_mode equal for now")

    # optional subsample per client (for speed)
    if args.subset_per_client and args.subset_per_client > 0:
        splits = [idxs[:args.subset_per_client] for idxs in splits]

    for i, idxs in enumerate(splits):
        print(f"[Split] client {i}: {len(idxs)} images", flush=True)

    # === Build CL schedule: 1 initial batch + (cl_batches-1) increments ===
    def make_cl_batches(indices, num_batches=7, seed=42):
        rng = np.random.RandomState(seed)
        idx = np.array(indices, dtype=np.int64)
        rng.shuffle(idx)

        # initial batch ~ 0.466 of data (e.g., 5250/11250 in your example)
        init = int(round(0.466 * len(idx)))
        init = max(1, min(len(idx) - (num_batches - 1), init))  # keep space for the increments

        first = idx[:init]
        rem = idx[init:]

        # split remaining evenly over (num_batches - 1)
        if num_batches <= 1:
            return [idx.tolist()]

        per = len(rem) // (num_batches - 1)
        chunks = [rem[i*per:(i+1)*per] for i in range(num_batches - 2)]
        chunks.append(rem[(num_batches - 2)*per:])  # last chunk gets the remainder
        return [first.tolist()] + [c.tolist() for c in chunks]


    # Build a per-client list of batches
    cl_schedule = []
    for cid, idxs in enumerate(splits):
        batches = make_cl_batches(idxs, num_batches=args.cl_batches, seed=args.seed + cid)
        cl_schedule.append(batches)
        sizes = [len(b) for b in batches]
        print(f"[CL] client {cid}: {sizes} (sum={sum(sizes)})", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_logs = []       # per-epoch logs
    round_logs = []     # per-round global acc

    # store CL schedule for export
    cl_rows = []
    for cid, batches in enumerate(cl_schedule):
        for i, b in enumerate(batches, start=1):
            cl_rows.append({
                "run_id": run_id,
                "client": cid,
                "cl_batch": i,
                "size": len(b)
            })
    


    # Optional: align rounds to cl_batches if you want one CL batch per FL round
    if args.rounds != args.cl_batches:
        print(f"[Note] args.rounds ({args.rounds}) != cl_batches ({args.cl_batches}). "
            f"If you want 1 batch per round, set --rounds {args.cl_batches}.", flush=True)

    # --- init clients ---
    # init clients
    clients = []
    for cid, idx in enumerate(splits):
        subset = Subset(trainset_full, idx)  # note: from *full* train with tf_train
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        model = build_resnet18(100).to(device)
        if args.optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        else:
            opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        replay = ReplayBuffer(capacity=2000)
        clients.append(Client(cid, model, opt, loader, device=device, replay=replay,
        val_loader=val_loader, early_patience=args.early_patience))
        

         ####### for testing: count params
        if cid == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable:,}/{total}", flush=True)

    # ✅ Check device
    print("✅ Client 0 model device:", next(clients[0].model.parameters()).device, flush=True)


    # --- server, policy, and initial eval ---
    server = Server(device=device)
    #policy = Policy()
    best_recall = np.zeros(100, dtype=np.float32)
    last_acc = None

    # initial global model (FedAvg of untrained client models)
    global_model = server.average([c.model for c in clients])
    acc, per_class = evaluate(global_model, device, test_loader)
    forgetting = np.maximum(0.0, best_recall - per_class)
    best_recall = np.maximum(best_recall, per_class)
    print(f"[Round -1] acc={acc:.3f}", flush=True)

    # --- training rounds ---
    for r in range(args.rounds):

        acc_delta = 0.0 if last_acc is None else (acc - last_acc)

        # Fixed hyperparameters (baseline CL setup)
        hp = {
            "lr": args.lr,
            "replay_ratio": 0.50,
            "notes": "fixed baseline"
        }

        print(
            f"[Round {r}] acc={acc:.3f} Δ={acc_delta:+.3f} "
            f"-> lr={hp['lr']:.5f}, replay={hp['replay_ratio']:.2f}",
            flush=True
        )

        # broadcast global + set lr
        for c in clients:
            c.load_state_from(global_model)
            for pg in c.optimizer.param_groups:
                pg["lr"] = hp["lr"]

        # --- local continual-learning training ---
        for c in clients:
            # Select the current CL batch for this round
            batches = cl_schedule[c.cid]
            if r < len(batches):
                batch_indices = batches[r]
                batch_id = r
            else:
                batch_indices = batches[-1]
                batch_id = len(batches) - 1

            # Assign new dataloader for this CL batch (incremental data)
            c.loader = DataLoader(
                Subset(trainset_full, batch_indices),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            print(f"[Round {r}] client {c.cid}: CL batch {batch_id+1}/{len(batches)} "
                f"(new={len(batch_indices)}; replay ratio≈{hp['replay_ratio']:.2f})",
                flush=True)

            # Train for several epochs, mixing replay from previous batches
            for e in range(args.epochs):     
                avg_loss, epoch_acc, stop = c.train_one_epoch(
                    replay_ratio=hp["replay_ratio"],
                    epoch=e,
                    total_epochs=args.epochs,
                    log_interval=args.log_interval,
                )
                # per-epoch log row
                run_logs.append({
                    "run_id": run_id,
                    "tag": getattr(args, "tag", "controller"),
                    "round": r,
                    "client": c.cid,
                    "epoch": e + 1,
                    "lr": hp["lr"],
                    "replay_ratio": hp["replay_ratio"],
                    "cl_batch": batch_id + 1,
                    "cl_batch_size": len(batch_indices),
                    "train_loss": float(avg_loss),
                    "train_acc": float(epoch_acc),
                    "val_loss": float(getattr(c, "_last_vloss", float("nan"))),
                    "val_acc": float(getattr(c, "_last_vacc", float("nan"))),
                })

                if stop:
                    print(
                        f"[Client {c.cid}] Early stopping triggered "
                        f"(no val improvement {c.early_patience} epochs).",
                        flush=True,
                    )
                    break


        # --- aggregate & evaluate global model ---
        global_model = server.average([c.model for c in clients])
        last_acc = acc
        acc, per_class = evaluate(global_model, device, test_loader)
        round_logs.append({
            "run_id": run_id,
            "tag": getattr(args, "tag", "controller"),
            "round": r,
            "global_acc": float(acc),
            "lr": hp["lr"],
            "replay_ratio": hp["replay_ratio"],
        })
        forgetting = np.maximum(0.0, best_recall - per_class)
        best_recall = np.maximum(best_recall, per_class)
        print(f"[Round {r}] acc={acc:.3f}", flush=True)

    # === write CSVs ===
    pd.DataFrame(run_logs).to_csv(f"fcl_run_results_controller_{run_id}.csv", index=False)
    pd.DataFrame(round_logs).to_csv(f"fcl_run_summary_controller_{run_id}.csv", index=False)
    pd.DataFrame(cl_rows).to_csv(f"fcl_run_cl_batches_controller_{run_id}.csv", index=False)
    print("✓ Wrote CSVs:",
        f"fcl_run_results_controller_{run_id}.csv,",
        f"fcl_run_summary_controller_{run_id}.csv,",
        f"fcl_run_cl_batches_controller_{run_id}.csv")
    
    # --- save checkpoint for Assignment 3 demo ---
    os.makedirs("assignment3/results/runs/latest", exist_ok=True)
    torch.save(global_model.state_dict(), "assignment3/results/runs/latest/global_last.pth")
    print("✓ Saved checkpoint: assignment3/results/runs/latest/global_last.pth")

if __name__ == "__main__":

    main()