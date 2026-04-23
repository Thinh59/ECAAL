"""
train.py — Training loop cho ablation study.

BUG ĐÃ SỬA (so với master gốc):
  1. loss_cfg.pop('name') làm MUT config dict → lần chạy thứ 2 trong cùng
     process sẽ không tìm được 'name'. Fix: dùng copy trước khi pop.
  2. torch.cuda.amp.autocast() deprecated trong PyTorch ≥ 2.0
     → dùng torch.amp.autocast('cuda') thay thế
  3. torch.cuda.amp.GradScaler() → torch.amp.GradScaler('cuda')

QUAN TRỌNG — evaluate.py FIX:
  evaluate_model() phải chạy fp32 (KHÔNG autocast) để sigmoid precision
  cao → mAP chính xác. Đây là nguyên nhân master cho mAP ~0.56 thay vì ~0.75.
"""

import os, sys, yaml, argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models   import build_model
from losses   import get_loss
from dataset  import get_dataloaders
from evaluate import evaluate_model
from utils    import AverageMeter, Logger, save_checkpoint, set_seed


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device, scaler, max_norm=1.0, consistency_alpha=0.0, sparsity_lambda=0.0):
    model.train()
    meter = AverageMeter()

    for i, (imgs, targets) in enumerate(loader):
        imgs    = imgs.to(device,    non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # AMP forward (PyTorch ≥ 2.0 style)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, targets)

            # --- Thí nghiệm 2: Consistency Alignment ---
            if consistency_alpha > 0.0 and getattr(model, 'use_cbam', False):
                # Lấy attention map của ảnh gốc
                att_orig = model.cbam.spatial_att.last_scale
                
                # Lật ngang ảnh, forward để lấy attention map lật
                imgs_flipped = torch.flip(imgs, dims=[3])
                _ = model(imgs_flipped)
                att_flipped = model.cbam.spatial_att.last_scale
                
                # Lật ngược attention map lại để so với bản gốc
                att_flipped_back = torch.flip(att_flipped, dims=[3])
                loss_cons = torch.nn.functional.mse_loss(att_flipped_back, att_orig)
                
                loss = loss + consistency_alpha * loss_cons

            # --- Thí nghiệm 3: Sparsity Constraints ---
            if sparsity_lambda > 0.0 and getattr(model, 'use_cbam', False):
                att_map = model.cbam.spatial_att.last_scale
                loss_sparse = torch.mean(torch.abs(att_map))
                loss = loss + sparsity_lambda * loss_sparse

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        meter.update(loss.item(), imgs.size(0))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(loader)}] loss={meter.avg:.4f}")

    return meter.avg


def run(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_name = Path(config_path).stem
    out_dir  = Path(cfg.get('output_dir', 'outputs')) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}\n[Exp] {exp_name} | device={device}\n{'='*55}")

    # Data
    train_loader, val_loader = get_dataloaders(cfg['data'])

    # Model
    model = build_model(cfg['model']).to(device)

    # Loss — FIX: copy dict để tránh mutate cfg khi pop 'name'
    loss_cfg  = dict(cfg.get('loss', {'name': 'asl'}))
    loss_name = loss_cfg.pop('name')
    criterion = get_loss(loss_name, **loss_cfg)
    print(f"[Loss] {loss_name.upper()} | params: {loss_cfg}")

    # Optimizer
    opt = cfg.get('optimizer', {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.get('lr', 3e-4),
        weight_decay=opt.get('weight_decay', 1e-4),
    )

    # Scheduler
    n_epochs = cfg.get('num_epochs', 20)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=opt.get('lr', 3e-4),
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1,
    )

    # AMP scaler (PyTorch ≥ 2.0)
    scaler = torch.amp.GradScaler('cuda')

    max_norm = cfg.get('max_norm', 1.0)
    logger   = Logger(str(out_dir / 'log.json'))
    best_map = 0.0

    consistency_alpha = cfg.get('train', {}).get('consistency_alpha', 0.0)
    sparsity_lambda = cfg.get('train', {}).get('sparsity_lambda', 0.0)

    for epoch in range(1, n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{n_epochs} ---")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, scaler, max_norm=max_norm,
            consistency_alpha=consistency_alpha, sparsity_lambda=sparsity_lambda
        )

        # Validate mỗi 2 epoch (tiết kiệm thời gian Kaggle)
        if epoch % 2 == 0 or epoch == n_epochs:
            metrics = evaluate_model(model, val_loader, device)
            metrics['train_loss'] = train_loss
            logger.log(epoch, metrics)
            logger.print_latest()

            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                save_checkpoint(model, optimizer, epoch, metrics,
                                str(out_dir / 'best.pth'))
                print(f"  ✅ Best mAP: {best_map:.4f}")
        else:
            print(f"  train_loss={train_loss:.4f}")

    print(f"\n[Done] {exp_name} | Best mAP={best_map:.4f}")
    return best_map


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    run(p.parse_args().config)
