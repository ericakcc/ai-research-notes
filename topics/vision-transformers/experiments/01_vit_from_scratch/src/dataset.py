"""CIFAR-10 dataset loading with augmentation."""

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import ViTConfig


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transform() -> transforms.Compose:
    """Training transforms with augmentation."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_dataloaders(
    config: ViTConfig,
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        config: ViT configuration (unused currently, reserved for future).
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        data_dir: Directory to download/store CIFAR-10.
        val_split: Fraction of training data for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_train_transform()
    )

    # Split training set into train/val
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    # Override transform for validation split (no augmentation)
    val_dataset.dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=get_val_transform()
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_val_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
