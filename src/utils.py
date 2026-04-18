"""utils.py — AverageMeter, Logger, Checkpoint, Seed."""

import os, json, time
import torch
import numpy as np


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


class Logger:
    def __init__(self, path: str):
        self.path    = path
        self.records = []

    def log(self, epoch: int, metrics: dict):
        self.records.append({'epoch': epoch, 'ts': time.time(), **metrics})
        with open(self.path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def print_latest(self):
        if not self.records: return
        r = self.records[-1]
        parts = [f"{k}={v:.4f}" for k, v in r.items()
                 if k not in ('epoch', 'ts') and isinstance(v, float)]
        print(f"[Epoch {r['epoch']}] " + " | ".join(parts))


def save_checkpoint(model, optimizer, epoch, metrics, path: str):
    torch.save({'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'metrics': metrics}, path)
    print(f"[Ckpt] Saved → {path}")


def load_checkpoint(model, optimizer, path: str, device: str = 'cuda'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
