import os
from monai.apps import DecathlonDataset
from monai.data import DataLoader

def get_datasets(config, train_transform, val_transform):
    task_folder = os.path.join(config.dataset_dir, "Task01_BrainTumour", "imagesTr")
    should_download = not os.path.exists(task_folder) or not os.listdir(task_folder)
    print("---------Config-------",config.num_workers)
    print("---------train_transform-------",train_transform)
    print("---------val_transform-------",val_transform)
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

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=config.num_workers,
    # )

    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=config.num_workers,
    # )
    print("Type of train_dataset:", type(train_dataset))
    return train_dataset, val_dataset


def get_val_dataset(config, transform):
    return DecathlonDataset(
        root_dir=config.dataset_dir,
        task="Task01_BrainTumour",
        transform=transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )