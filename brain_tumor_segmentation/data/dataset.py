import os

import dvc.api
from monai.apps import DecathlonDataset


def download_data(config):
    """Download dataset using DVC."""
    dataset_path = os.path.join(config.dataset_dir, "Task01_BrainTumour")
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("Pulling dataset with DVC...")
        dvc.api.get_url(
            url="dvc.yaml",  # Adjust to your DVC configuration
            repo=".",  # Local or remote repository
            path=dataset_path,
        )
    else:
        print("Dataset already exists locally.")


def get_datasets(config, train_transform, val_transform):
    download_data(config)
    task_folder = os.path.join(config.dataset_dir, "Task01_BrainTumour", "imagesTr")
    should_download = not os.path.exists(task_folder) or not os.listdir(task_folder)
    print("---------Config-------", config.num_workers)
    print("---------train_transform-------", train_transform)
    print("---------val_transform-------", val_transform)
    train_dataset = DecathlonDataset(
        root_dir=config.dataset_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=should_download,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )

    val_dataset = DecathlonDataset(
        root_dir=config.dataset_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )

    print("Type of train_dataset:", type(train_dataset))
    return train_dataset, val_dataset


def get_val_dataset(config, transform):
    download_data(config)
    return DecathlonDataset(
        root_dir=config.dataset_dir,
        task="Task01_BrainTumour",
        transform=transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )
