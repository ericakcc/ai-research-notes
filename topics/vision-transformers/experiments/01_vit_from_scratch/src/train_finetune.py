"""Track B: ViT-Small fine-tuned from ImageNet-21k pretrained weights on CIFAR-10.

Comparison experiment for pretrain_vs_scratch.
Architecture: timm vit_small_patch16_224, pretrained=True.
Demonstrates ViT paper Section 3.2: pretrained ViT outperforms scratch ViT on small datasets.
"""

import argparse
from pathlib import Path

import timm
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_cifar10_224_loaders


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float = 0.75,
) -> AdamW:
    """Build AdamW optimizer with layer-wise learning rate decay (LLRD).

    Shallower layers receive a smaller lr to preserve pretrained features.
    The classification head receives the full lr.

    Args:
        model: timm ViT model with .blocks attribute.
        lr: Base learning rate (applied to head).
        weight_decay: L2 regularization strength.
        layer_decay: Multiplicative lr decay per layer away from head.

    Returns:
        AdamW optimizer with per-parameter lr.
    """
    num_layers = len(model.blocks) + 1  # blocks 0..11, then head = layer 12

    def get_layer_id(name: str) -> int:
        if name.startswith(("patch_embed", "pos_embed", "cls_token")):
            return 0
        if name.startswith("blocks"):
            return int(name.split(".")[1]) + 1
        return num_layers  # head, norm

    param_groups = [
        {
            "params": [param],
            "lr": lr * (layer_decay ** (num_layers - get_layer_id(name))),
        }
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    return AdamW(param_groups, weight_decay=weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: timm ViT model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Args:
        model: timm ViT model.
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


def main() -> None:
    """Main training loop for Track B (pretrained fine-tune)."""
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained ViT-Small on CIFAR-10"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    pin_memory = device.type == "cuda"
    print(f"[Track B] Pretrained Fine-tune | Device: {device}")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project="vit-cifar10",
            name="track-b-finetune",
            config=vars(args),
        )

    model = timm.create_model(
        "vit_small_patch16_224", pretrained=True, num_classes=10
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: vit_small_patch16_224 (ImageNet-21k pretrained)")
    print(f"Parameters: {num_params:,}")

    train_loader, val_loader, _ = get_cifar10_224_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir, pin_memory=pin_memory
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"\n[Track B] Training complete. Best val accuracy: {best_val_acc:.4f}")
    if use_wandb:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.finish()


if __name__ == "__main__":
    main()
