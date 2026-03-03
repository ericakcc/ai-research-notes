"""ViT ablation study: progressive training tricks on CIFAR-10.

Tracks:
    A (baseline) : AdamW + CosineAnnealingLR + RandomCrop + RandomFlip
    B (+warmup)  : A + LR warmup 10 epochs + gradient clipping 1.0
    C (+smooth)  : B + label smoothing 0.1
    D (+randaug) : C + RandAugment(2, 9)
    E (+mixup)   : D + Mixup alpha=0.8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import ViTConfig
from dataset import get_dataloaders, get_train_transform, get_val_transform
from model import VisionTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACKS: dict[str, dict[str, object]] = {
    "baseline": {},
    "warmup": {"warmup_epochs": 10, "grad_clip": 1.0},
    "smooth": {"warmup_epochs": 10, "grad_clip": 1.0, "label_smoothing": 0.1},
    "randaug": {
        "warmup_epochs": 10,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "use_randaugment": True,
    },
    "full": {
        "warmup_epochs": 10,
        "grad_clip": 1.0,
        "label_smoothing": 0.1,
        "use_randaugment": True,
        "mixup_alpha": 0.8,
    },
}


@dataclass
class TrainConfig:
    """Training hyper-parameters and trick toggles.

    Args:
        epochs: Total training epochs.
        batch_size: Mini-batch size.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear warmup duration (0 = disabled).
        grad_clip: Max gradient norm (0 = disabled).
        label_smoothing: Label smoothing epsilon (0 = disabled).
        use_randaugment: Whether to apply RandAugment.
        mixup_alpha: Mixup Beta distribution alpha (0 = disabled).
    """

    epochs: int = 100
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 0
    grad_clip: float = 0.0
    label_smoothing: float = 0.0
    use_randaugment: bool = False
    mixup_alpha: float = 0.0
    wandb_project: str = "vit-cifar10"


# ---------------------------------------------------------------------------
# Mixup helper
# ---------------------------------------------------------------------------


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup to a batch of images and labels.

    Args:
        images: Input images (B, C, H, W).
        labels: Integer class labels (B,).
        alpha: Beta distribution parameter.
        num_classes: Number of classes for one-hot conversion.

    Returns:
        Tuple of (mixed_images, mixed_labels) where mixed_labels are soft
        probability vectors of shape (B, num_classes).
    """
    # TODO(human): Implement mixup logic (~5-8 lines)
    #
    # Steps:
    #   1. Sample lambda from Beta(alpha, alpha)
    #   2. Generate a random permutation of the batch indices
    #   3. Mix images:  mixed = lam * images + (1 - lam) * images[perm]
    #   4. Convert integer labels to one-hot vectors (use F.one_hot)
    #   5. Mix labels:  mixed = lam * labels_oh + (1 - lam) * labels_oh[perm]
    #   6. Return (mixed_images, mixed_labels)
    #
    # Available: torch.distributions.Beta, F.one_hot, torch.randperm
    lam = torch.distributions.Beta(alpha, alpha).sample()
    lam = max(lam.item(), 1.0 - lam.item())
    perm = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    labels_oh = F.one_hot(labels, num_classes).float()
    mixed_labels = lam * labels_oh + (1.0 - lam) * labels_oh[perm]
    return mixed_images, mixed_labels


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: VisionTransformer,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float = 0.0,
    mixup_alpha: float = 0.0,
    num_classes: int = 10,
) -> tuple[float, float]:
    """Train for one epoch with optional tricks.

    Args:
        model: ViT model.
        loader: Training data loader.
        criterion: Loss function (may include label smoothing).
        optimizer: Optimizer.
        device: Device to run on.
        grad_clip: Max gradient norm (0 = no clipping).
        mixup_alpha: Mixup alpha (0 = no mixup).
        num_classes: Number of classes (for mixup one-hot).

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        # --- Mixup ---
        if mixup_alpha > 0.0:
            images, mixed_labels = mixup_data(
                images, labels, mixup_alpha, num_classes
            )
            logits = model(images)
            # Cross-entropy with soft labels
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(mixed_labels * log_probs).sum(dim=1).mean()
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()

        # --- Gradient clipping ---
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        # For accuracy, use argmax even when using mixup
        preds = logits.argmax(dim=1)
        if mixup_alpha > 0.0:
            correct += (preds == mixed_labels.argmax(dim=1)).sum().item()
        else:
            correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: VisionTransformer,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: ViT model.
        loader: Data loader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Scheduler builder
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler with optional linear warmup + cosine decay.

    Args:
        optimizer: The optimizer.
        total_epochs: Total training epochs.
        warmup_epochs: Warmup duration (0 = pure cosine).

    Returns:
        A learning rate scheduler.
    """
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=total_epochs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_track(
    track_name: str,
    cfg: TrainConfig,
    data_dir: str,
    save_dir: str,
    *,
    use_wandb: bool = True,
) -> float:
    """Run a single ablation track.

    Args:
        track_name: Name of the track for logging.
        cfg: Training configuration.
        data_dir: CIFAR-10 data directory.
        save_dir: Checkpoint save directory.
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        Best validation accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Track: {track_name}")
    print(f"Config: {cfg}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # W&B
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=f"ablation-{track_name}",
            config={
                "track": track_name,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "warmup_epochs": cfg.warmup_epochs,
                "grad_clip": cfg.grad_clip,
                "label_smoothing": cfg.label_smoothing,
                "use_randaugment": cfg.use_randaugment,
                "mixup_alpha": cfg.mixup_alpha,
            },
        )

    # Model
    vit_config = ViTConfig()
    model = VisionTransformer(vit_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        vit_config,
        batch_size=cfg.batch_size,
        data_dir=data_dir,
        use_randaugment=cfg.use_randaugment,
    )

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
    )

    # Training loop
    out_dir = Path(save_dir) / track_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=cfg.grad_clip,
            mixup_alpha=cfg.mixup_alpha,
            num_classes=vit_config.num_classes,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{track_name}] Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": current_lr,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> New best (val_acc={val_acc:.4f})")

    print(f"\n[{track_name}] Done. Best val accuracy: {best_val_acc:.4f}")
    if use_wandb:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.finish()
    return best_val_acc


def main() -> None:
    """Run ablation experiments."""
    parser = argparse.ArgumentParser(description="ViT ablation study")
    parser.add_argument(
        "--track",
        type=str,
        default="baseline",
        choices=[*TRACKS, "all"],
        help="Which track to run (or 'all').",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging."
    )
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    tracks_to_run = list(TRACKS) if args.track == "all" else [args.track]
    results: dict[str, float] = {}

    for name in tracks_to_run:
        overrides = TRACKS[name]
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            **overrides,  # type: ignore[arg-type]
        )
        best = run_track(
            name, cfg, args.data_dir, args.save_dir, use_wandb=use_wandb
        )
        results[name] = best

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("Ablation Summary")
        print(f"{'='*60}")
        for name, acc in results.items():
            print(f"  {name:<12s} : {acc:.4f}")


if __name__ == "__main__":
    main()
