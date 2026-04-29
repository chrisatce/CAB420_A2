"""
    training_segformer.py  –  SegFormer trainer for the DroneDeploy benchmark

    Changes vs original:
        • Tracks train/val loss AND pixel-accuracy each epoch
        • Returns (model, history) so the caller can inspect or save metrics
        • Saves training curves to  plots/training_curves.png  automatically
        • Accepts optional pretrained_weights path for fine-tuning
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

import matplotlib
matplotlib.use('Agg')          # headless – no display needed
import matplotlib.pyplot as plt

from libs.models_segformer import SegFormer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChipDataset(Dataset):
    def __init__(self, dataset, split='train', size=300):
        self.size       = size
        self.image_dir  = f'{dataset}/image-chips'
        self.label_dir  = f'{dataset}/label-chips'

        with open(f'{dataset}/{split}.txt') as f:
            self.files = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img = np.array(
            Image.open(f'{self.image_dir}/{fname}').convert('RGB')
        ).astype(np.float32) / 255.0

        lbl = np.array(
            Image.open(f'{self.label_dir}/{fname}')
        )[:, :, 0].astype(np.int64)

        img = torch.from_numpy(img).permute(2, 0, 1)   # (C, H, W)
        lbl = torch.from_numpy(lbl)                    # (H, W)
        return img, lbl


# ---------------------------------------------------------------------------
# Pixel accuracy helper  (ignores class 0 = IGNORE, matching scoring.py)
# ---------------------------------------------------------------------------

def pixel_accuracy(preds_logits, labels, ignore_index=0):
    """
    preds_logits : (B, C, H, W) raw logits
    labels       : (B, H, W)   int class indices
    Returns fraction of non-ignored pixels classified correctly.
    """
    preds = preds_logits.argmax(dim=1)          # (B, H, W)
    mask  = labels != ignore_index
    if mask.sum() == 0:
        return float('nan')
    correct = ((preds == labels) & mask).sum().item()
    total   = mask.sum().item()
    return correct / total


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def save_training_curves(history, savedir='plots'):
    """
    Saves a 2-panel figure:
        Left  – Train vs Val Loss
        Right – Train vs Val Pixel Accuracy
    """
    os.makedirs(savedir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-o', markersize=4, label='Train Loss')
    ax.plot(epochs, history['val_loss'],   'r-o', markersize=4, label='Val Loss')
    ax.set_title('Loss per Epoch', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    train_acc = [a * 100 for a in history['train_acc']]
    val_acc   = [a * 100 for a in history['val_acc']]
    ax.plot(epochs, train_acc, 'b-o', markersize=4, label='Train Acc')
    ax.plot(epochs, val_acc,   'r-o', markersize=4, label='Val Acc')
    ax.set_title('Pixel Accuracy per Epoch', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pixel Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('SegFormer Training Curves', fontsize=15, fontweight='bold')
    fig.tight_layout()

    savepath = os.path.join(savedir, 'training_curves.png')
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  ↳ training curves saved → {savepath}')
    return savepath


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train_model(
    dataset,
    variant            = 'B2',
    epochs             = 1,
    lr                 = 6e-5,
    bs                 = 2,
    size               = 300,
    weights            = None,
    pretrained_weights = None,
    accumulate_steps   = 2,
    patience           = 5,
):
    """
    Train a SegFormer model and return (model, history).

    history is a dict with keys:
        train_loss, val_loss, train_acc, val_acc
    each a list of length `epochs`.
    """
    # Accept weights from either arg name
    ckpt = pretrained_weights or weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n==============================")
    print(f"  Training SegFormer-{variant}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {bs}  (accumulate × {accumulate_steps} = eff. bs {bs * accumulate_steps})")
    print("==============================\n")

    # ---- data ---------------------------------------------------------------
    train_ds = ChipDataset(dataset, 'train', size)
    valid_ds = ChipDataset(dataset, 'valid', size)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=0, pin_memory=False)
    valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                          num_workers=0, pin_memory=False)

    # ---- model --------------------------------------------------------------
    model = SegFormer(
        num_classes  = 6,
        variant      = variant,
        decoder_dim  = 256 if variant in ('B0', 'B1') else 768,
        output_size  = (size, size),
    ).to(device)

    if ckpt and os.path.exists(ckpt):
        print(f"  Loading weights: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        # Support loading encoder-only ImageNet weights
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  (missing keys: {len(missing)} – decoder will train from scratch)")
        except Exception as e:
            print(f"  Weight load warning: {e}")

    # ---- optimiser ----------------------------------------------------------
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': lr * 0.1},
        {'params': model.decoder.parameters(), 'lr': lr},
    ], weight_decay=1e-2)

    scheduler = OneCycleLR(
        optimizer,
        max_lr      = [lr * 0.1, lr],
        total_steps = epochs * len(train_dl),
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    scaler  = GradScaler()

    os.makedirs('models', exist_ok=True)

    # ---- history bookkeeping ------------------------------------------------
    history = {
        'train_loss': [],
        'val_loss'  : [],
        'train_acc' : [],
        'val_acc'   : [],
    }

    best_val_loss  = float('inf')
    patience       = 5    # stop if no improvement for this many epochs
    patience_count = 0

    # ---- training loop ------------------------------------------------------
    for epoch in range(epochs):

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_acc  = 0.0
        optimizer.zero_grad()

        for i, (imgs, lbls) in enumerate(train_dl):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            with autocast():
                out  = model(imgs)
                loss = loss_fn(out, lbls) / accumulate_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * accumulate_steps
            with torch.no_grad():
                running_acc += pixel_accuracy(out.detach(), lbls)

        train_loss = running_loss / len(train_dl)
        train_acc  = running_acc  / len(train_dl)

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0

        with torch.no_grad():
            for imgs, lbls in valid_dl:
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                with autocast():
                    out  = model(imgs)
                    loss = loss_fn(out, lbls)

                val_loss += loss.item()
                val_acc  += pixel_accuracy(out, lbls)

        val_loss /= len(valid_dl)
        val_acc  /= len(valid_dl)

        # ── record ─────────────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:03d}/{epochs}"
              f"  |  train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%"
              f"  |  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        # ── save best / early stopping ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            ckpt_path = f"models/segformer_{variant.lower()}_best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"         ↳ saved best model  (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            print(f"         ↳ no improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} — no improvement for {patience} epochs.")
                break

    # ---- save plots ---------------------------------------------------------
    save_training_curves(history)

    return model, history