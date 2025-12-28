# src/fl.py
from typing import List, Optional
from copy import deepcopy
import torch
from torch import nn
from assignment3.src.strategies.replay import ReplayBuffer


class Server:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")

    def average(self, models: List[nn.Module]) -> nn.Module:
        # FedAvg on CPU or given device
        avg = deepcopy(models[0]).to(self.device)
        with torch.no_grad():
            for k in avg.state_dict().keys():
                s = sum(m.state_dict()[k].to(self.device) for m in models) / len(models)
                avg.state_dict()[k].copy_(s)
        return avg.to(self.device)

class Client:
    def __init__(self, cid: int, model: nn.Module, optimizer, train_loader,
                 device: Optional[torch.device] = None, replay: Optional[ReplayBuffer] = None,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 early_patience: int = 5):
        self.cid = cid
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)                    
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.loader = train_loader
        self.replay = replay or ReplayBuffer(capacity=2000)
        self.val_loader = val_loader
        self.early_patience = early_patience
        self._best_val = None
        self._no_improve = 0


    def load_state_from(self, global_model: nn.Module):
        # load weights then keep model on device
        self.model.load_state_dict(global_model.state_dict())
        self.model.to(self.device)    
        self._best_val = None
        self._no_improve = 0  

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss_sum += float(loss.item())
            pred = logits.argmax(1)
            total += y.numel()
            correct += (pred == y).sum().item()
        avg_loss = loss_sum / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        return avg_loss, acc                      

    def train_one_epoch(
        self,
        replay_ratio: float = 0.2,
        epoch: int = 0,
        total_epochs: int = 1,
        log_interval: int = 200,
    ):
        self.model.train()
        num_batches = len(self.loader)
        running_loss = 0.0
        correct = 0
        total = 0

        for b, (x, y) in enumerate(self.loader, 1):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            # optional replay
            if self.replay is not None and replay_ratio > 0.0:
                rx, ry = self.replay.sample_like(x.size(0), device=self.device, ratio=replay_ratio)
                if rx is not None:
                    rx = rx.to(self.device, non_blocking=True)
                    ry = ry.to(self.device, non_blocking=True)
                    x = torch.cat([x, rx], dim=0)
                    y = torch.cat([y, ry], dim=0)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())

            # accuracy on current (possibly replay-mixed) batch
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                batch_correct = (preds == y).sum().item()
                batch_total = y.size(0)
                correct += batch_correct
                total += batch_total

            # store CPU copies for replay
            if self.replay is not None:
                self.replay.add_batch(x.detach().cpu(), y.detach().cpu())

            # periodic log
            if (b % log_interval) == 0 or b == num_batches:
                print(
                    f"[Client {self.cid}] epoch {epoch+1}/{total_epochs} "
                    f"batch {b}/{num_batches} loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = running_loss / max(1, num_batches)
        epoch_acc = 100.0 * correct / max(1, total)

        # --- Validation + early stopping ---
        val_note = ""
        if self.val_loader is not None:
            vloss, vacc = self.evaluate(self.val_loader)
            self._last_vloss = vloss
            self._last_vacc  = vacc
            val_note = f", val_loss={vloss:.4f}, val_acc={vacc:.2f}%"
            improved = (self._best_val is None) or (vacc > self._best_val + 1e-4)
            if improved:
                self._best_val = vacc
                self._no_improve = 0
            else:
                self._no_improve += 1

        print(
            f"[Client {self.cid}] epoch {epoch+1}/{total_epochs} done, "
            f"avg_loss={avg_loss:.4f}, epoch_acc={epoch_acc:.2f}%{val_note}",
            flush=True,
        )

        should_stop = (self.val_loader is not None) and (self._no_improve >= self.early_patience)
        return avg_loss, epoch_acc, should_stop