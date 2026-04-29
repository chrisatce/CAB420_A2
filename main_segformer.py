"""
    main_segformer.py  –  SegFormer pipeline for the DroneDeploy benchmark

    Drop this into the root of dd-ml-segmentation-benchmark-master.

    Usage:
        python main_segformer.py
        python main_segformer.py --variant B0
        python main_segformer.py --variant B2 --dataset dataset-medium
        python main_segformer.py --variant B2 --weights models/mit_b2.pth

    Pretrained MiT encoder weights (ImageNet-1K):
        https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5
        Download mit_b2.pth (or the variant you want) and pass via --weights.
        This dramatically improves segmentation quality.

    Outputs:
        predictions/          –  per-scene colour-mask PNGs
        predictions/*.png     –  confusion-matrix PNGs (from scoring.py)
        models/segformer_*_best.pth
        plots/training_curves.png
        results.json
"""

import argparse
import os
import json
import time

import numpy as np
import torch

from PIL import Image, ImageFile

# Safe loading for large / truncated GeoTiffs
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import libs.images2chips
from libs import scoring
from libs.training_segformer import train_model
from libs.config import test_ids, LABELMAP


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='SegFormer segmentation pipeline')
    p.add_argument('--variant', default='B0',
                   choices=['B0', 'B1', 'B2', 'B3', 'B4', 'B5'],
                   help='MiT encoder variant (B0=3.7M … B5=82M params)')
    p.add_argument('--dataset', default='dataset-sample',
                   choices=['dataset-sample', 'dataset-medium'])
    p.add_argument('--weights', default=None,
                   help='Optional path to pretrained MiT weights (.pth)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5,
                   help='Early stopping patience (default 5)')
    p.add_argument('--lr', type=float, default=6e-5)
    p.add_argument('--bs', type=int, default=2,
                   help='Per-GPU batch size (default 2 is VRAM-safe)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference  (batched sliding-window, pure PyTorch)
# ---------------------------------------------------------------------------

def run_inference(dataset, model, device, basedir='predictions', size=300):
    os.makedirs(basedir, exist_ok=True)
    model.eval()

    for scene in test_ids:
        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        predsfile = os.path.join(basedir, f'{scene}-prediction.png')

        if not os.path.exists(imagefile):
            continue

        print(f'  inference: {imagefile}')

        try:
            img = np.array(Image.open(imagefile).convert('RGB'))
        except Exception as e:
            print(f'  SKIP {imagefile}: {e}')
            continue

        h, w, _ = img.shape
        pred_map = np.zeros((h, w), dtype=np.uint8)

        chips, coords = [], []

        for x in range(0, w, size):
            for y in range(0, h, size):
                chip = img[y:y+size, x:x+size]
                if chip.size == 0:
                    continue
                pad_y = max(0, size - chip.shape[0])
                pad_x = max(0, size - chip.shape[1])
                if pad_y > 0 or pad_x > 0:
                    chip = np.pad(chip, [(0, pad_y), (0, pad_x), (0, 0)], mode='constant')
                chips.append(chip)
                coords.append((y, x))

        if not chips:
            continue

        # ---- batch → GPU (in mini-batches to avoid OOM) --------------------
        all_preds = []
        batch_size = 4  # reduce to 2 if still OOM
        for batch_start in range(0, len(chips), batch_size):
            batch = chips[batch_start:batch_start + batch_size]
            tensor = torch.tensor(
                np.stack(batch), dtype=torch.float32
            ).permute(0, 3, 1, 2).to(device) / 255.0
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    batch_preds = model(tensor).argmax(1).cpu().numpy()
            all_preds.append(batch_preds)
            del tensor
            torch.cuda.empty_cache()
        preds = np.concatenate(all_preds, axis=0)

        # ---- stitch prediction map -----------------------------------------
        # Model outputs 0-5; LABELMAP uses 1-6 for real classes (0 = IGNORE)
        for (y, x), p in zip(coords, preds):
            pred_map[y:y+size, x:x+size] = (p + 1)[:max(0, h-y), :max(0, w-x)]

        # ---- colourise and save (BGR — matches cv2.imread in scoring.py) ---
       # ---- colourise and save (BGR — matches cv2.imread in scoring.py) ---
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c, color in LABELMAP.items():
            mask[pred_map == c] = color
        import cv2
        cv2.imwrite(predsfile, mask)


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_full_stats(scores, history):
    """
    Pretty-print the complete evaluation suite:
        • Per-epoch best results
        • Final test-set scores (all six metrics)
    """
    print()
    print("╔══════════════════════════════════════════╗")
    print("║          SEGFORMER FINAL RESULTS          ║")
    print("╠══════════════════════════════════════════╣")

    # Training summary
    best_epoch = int(np.argmin(history['val_loss'])) + 1
    best_val   = min(history['val_loss'])
    best_acc   = history['val_acc'][best_epoch - 1] * 100
    final_acc  = history['val_acc'][-1] * 100

    print(f"║  Epochs trained      : {len(history['val_loss']):<18d}║")
    print(f"║  Best epoch          : {best_epoch:<18d}║")
    print(f"║  Best val loss       : {best_val:<18.4f}║")
    print(f"║  Best val accuracy   : {best_acc:<17.2f}%║")
    print(f"║  Final val accuracy  : {final_acc:<17.2f}%║")
    print("╠══════════════════════════════════════════╣")
    print("║          TEST SET SCORES                  ║")
    print("╠══════════════════════════════════════════╣")

    label_map = {
        'f1_mean': 'F1 (mean)',
        'f1_std' : 'F1 (std)',
        'pr_mean': 'Precision (mean)',
        'pr_std' : 'Precision (std)',
        're_mean': 'Recall (mean)',
        're_std' : 'Recall (std)',
    }
    for key, label in label_map.items():
        val = scores.get(key, float('nan'))
        print(f"║  {label:<22s}: {val:<14.4f}  ║")

    print("╚══════════════════════════════════════════╝")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print()
    print("══════════════════════════════════════════")
    print("  SegFormer Segmentation Pipeline")
    print("══════════════════════════════════════════")
    print(f"  Variant  : {args.variant}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Device   : {device}")
    if args.weights:
        print(f"  Weights  : {args.weights}")
    print("══════════════════════════════════════════\n")

    # ── 1. Chip preparation ─────────────────────────────────────────────────
    if not os.path.exists(f'{args.dataset}/image-chips') or \
       not os.path.exists(f'{args.dataset}/label-chips'):
        print("Preparing image chips…")
        libs.images2chips.run(args.dataset)
    else:
        print("Chips already exist, skipping.\n")

    # ── 2. Train ────────────────────────────────────────────────────────────
    t0 = time.time()
    model, history = train_model(
        dataset            = args.dataset,
        variant            = args.variant,
        epochs             = args.epochs,
        lr                 = args.lr,
        bs                 = args.bs,
        pretrained_weights = args.weights,
        patience           = args.patience,
    )
    train_time = time.time() - t0
    print(f"\n  Training wall-time: {train_time/60:.1f} min")

    model = model.to(device)

    # ── 3. Load best checkpoint for inference ───────────────────────────────
    best_ckpt = f"models/segformer_{args.variant.lower()}_best.pth"
    if os.path.exists(best_ckpt):
        print(f"\n  Loading best checkpoint for inference: {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # ── 4. Inference ────────────────────────────────────────────────────────
    print("\n  Running inference on test scenes…")
    t1 = time.time()
    run_inference(args.dataset, model, device)
    infer_time = time.time() - t1
    print(f"  Inference wall-time: {infer_time:.1f} s")

    # ── 5. Evaluation ───────────────────────────────────────────────────────
    print("\n  Scoring predictions…")
    scores, _ = scoring.score_predictions(args.dataset)

    # ── 6. Full statistics printout ─────────────────────────────────────────
    print_full_stats(scores, history)

    # ── 7. Save results JSON ────────────────────────────────────────────────
    results = {
        'variant'        : args.variant,
        'dataset'        : args.dataset,
        'train_time_min' : round(train_time / 60, 2),
        'infer_time_s'   : round(infer_time, 2),
        'scores'         : scores,
        'history'        : {
            'train_loss': [round(v, 5) for v in history['train_loss']],
            'val_loss'  : [round(v, 5) for v in history['val_loss']],
            'train_acc' : [round(v, 5) for v in history['train_acc']],
            'val_acc'   : [round(v, 5) for v in history['val_acc']],
        },
    }

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("  Saved → results.json")
    print("  Saved → plots/training_curves.png")
    print()