"""
train.py

Training Entry Point

User can choose one of two modes:

1) Train directly on FairFace images

2) Train on SAM2 extracted faces

Architecture

Image
   │
   ▼
Vision Encoder (Frozen)
   │
Projection
   │
├──────────────┐
│              │
▼              ▼
Gender Head    Age Head
               │
               ▼
           Race Head
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from main import load_fairface

from trainer import Trainer

from dataset import FaceDataset

from fairface_dataset import FairFaceDataset
############################################################
# Ask Dataset Mode
############################################################

def ask_dataset_mode():

    print()

    print("=" * 60)
    print("Dataset Selection")
    print("=" * 60)

    print("1) Original FairFace Images")

    print("2) SAM2 Cropped Faces")

    print("=" * 60)

    while True:

        choice = input("Select (1 or 2): ").strip()

        if choice == "1":

            return False

        elif choice == "2":

            return True

        print("Invalid choice.")

############################################################
# Check Processed Dataset
############################################################

def processed_dataset_exists():

    folders = [

        "processed_faces/train",

        "processed_faces/validation",

        "labels/train.csv",

        "labels/validation.csv"

    ]

    for item in folders:

        if not Path(item).exists():

            return False

    return True

############################################################
# Build SAM Dataset
############################################################

def build_sam_dataset():

    if not processed_dataset_exists():

        print()

        print("=" * 60)

        print("Processed dataset not found.")

        print("Running preprocess_dataset.py")

        print("=" * 60)

        import preprocess_dataset

        preprocess_dataset.main()

    train_dataset = FaceDataset(

        image_dir="processed_faces/train",

        csv_file="labels/train.csv",

        encoder_name="clip"

    )

    validation_dataset = FaceDataset(

        image_dir="processed_faces/validation",

        csv_file="labels/validation.csv",

        encoder_name="clip"

    )

    return train_dataset, validation_dataset

############################################################
# Build FairFace Dataset
############################################################

def build_fairface_dataset():

    train_data, test_data, validation_data = load_fairface()

    train_dataset = FairFaceDataset(

        train_data,

        encoder_name="clip"

    )

    validation_dataset = FairFaceDataset(

        validation_data,

        encoder_name="clip"

    )

    return train_dataset, validation_dataset

############################################################
# Create DataLoaders
############################################################

def create_dataloaders(

        train_dataset,

        validation_dataset,

        batch_size=16,

        num_workers=4

):

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available()

    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available()

    )

    return train_loader, validation_loader

############################################################
# Main
############################################################

def main():

    ########################################################
    # Ask User
    ########################################################

    use_sam = ask_dataset_mode()

    ########################################################
    # Dataset
    ########################################################

    print()
    print("=" * 70)
    print("Loading Dataset...")
    print("=" * 70)

    if use_sam:

        train_dataset, validation_dataset = build_sam_dataset()

        dataset_name = "SAM2 Cropped Faces"

    else:

        train_dataset, validation_dataset = build_fairface_dataset()

        dataset_name = "Original FairFace"

    ########################################################
    # DataLoader
    ########################################################

    train_loader, validation_loader = create_dataloaders(

        train_dataset=train_dataset,

        validation_dataset=validation_dataset,

        batch_size=16,

        num_workers=4

    )

    ########################################################
    # Dataset Information
    ########################################################

    print()

    print("=" * 70)

    print("Dataset Information")

    print("=" * 70)

    print(f"Dataset Mode      : {dataset_name}")

    print(f"Training Images   : {len(train_dataset)}")

    print(f"Validation Images : {len(validation_dataset)}")

    print()

    ########################################################
    # Trainer
    ########################################################

    trainer = Trainer(

        encoder_name="clip",

        model_path="../models/clip-vit-base-patch32",

        projection_dim=512,

        freeze_encoder=True,

        lr=1e-4,

        weight_decay=1e-4,

        gender_weight=1.0,

        age_weight=1.0,

        race_weight=1.0,

        checkpoint_dir="checkpoints",

        patience=5

    )

    ########################################################
    # Model Information
    ########################################################

    trainable_params = sum(

        p.numel()

        for p in trainer.model.parameters()

        if p.requires_grad

    )

    total_params = sum(

        p.numel()

        for p in trainer.model.parameters()

    )

    print()

    print("=" * 70)

    print("Model Information")

    print("=" * 70)

    print("Encoder           : CLIP")

    print("Frozen Encoder    : True")

    print("Projection Dim    : 512")

    print()

    print(f"Trainable Params  : {trainable_params:,}")

    print(f"Total Params      : {total_params:,}")

    print("=" * 70)

    ########################################################
    # Start Training
    ########################################################

    history = trainer.fit(

        train_loader=train_loader,

        validation_loader=validation_loader,

        epochs=20

    )

    ########################################################
    # Finished
    ########################################################

    print()

    print("=" * 70)

    print("Training Finished Successfully")

    print("=" * 70)

    print()

    print("Best model saved to:")

    print("checkpoints/best_model.pt")

    print()

    print("Last model saved to:")

    print("checkpoints/last_model.pt")

    print()

    print("History saved to:")

    print("checkpoints/history.csv")

    return history


############################################################
# Entry
############################################################

if __name__ == "__main__":

    main()