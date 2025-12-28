# src/strategies/replay.py
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.data = deque()

    def add_batch(self, x: torch.Tensor, y: torch.Tensor):
        # x: [B, C, H, W], y: [B]
        for i in range(x.size(0)):
            self.data.append((x[i].cpu(), y[i].cpu()))
            while len(self.data) > self.capacity:
                self.data.popleft()

    def sample_like(self, batch_size: int, device, ratio: float = 0.2):
        k = int(batch_size * ratio)
        if k <= 0 or len(self.data) == 0:
            return None, None
        k = min(k, len(self.data))
        samples = random.sample(self.data, k)
        xs = torch.stack([s[0] for s in samples]).to(device)
        ys = torch.stack([s[1] for s in samples]).to(device)
        return xs, ys