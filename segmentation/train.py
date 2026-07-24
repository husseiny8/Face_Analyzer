import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FaceDataset
from trainer import Trainer

############################################################
# Configuration
############################################################

# Dataset
TRAIN_IMAGE_DIR = "processed_faces/train"
TRAIN_LABEL_FILE = "labels/train.csv"

VALID_IMAGE_DIR = TRAIN_IMAGE_DIR
VALID_LABEL_FILE = TRAIN_LABEL_FILE

# Encoder
ENCODER_NAME = "clip"
MODEL_PATH = "../models/clip-vit-base-patch32"

# Training
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

FREEZE_ENCODER = True

PROJECTION_DIM = 512

CHECKPOINT_DIR = "checkpoints"

SEED = 42


############################################################
# Seed
############################################################

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


############################################################
# Count Parameters
############################################################

def count_parameters(model):

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    total = sum(

        p.numel()

        for p in model.parameters()

    )

    return trainable, total


############################################################
# Main
############################################################

def main():

    ########################################################
    # Random Seed
    ########################################################

    set_seed(SEED)

    ########################################################
    # Dataset
    ########################################################

    print("=" * 70)
    print("Loading Dataset...")
    print("=" * 70)

    train_dataset = FaceDataset(

        image_dir=TRAIN_IMAGE_DIR,

        csv_file=TRAIN_LABEL_FILE,

        encoder_name=ENCODER_NAME,

        processor_path=MODEL_PATH

    )

    validation_dataset = FaceDataset(

        image_dir=VALID_IMAGE_DIR,

        csv_file=VALID_LABEL_FILE,

        encoder_name=ENCODER_NAME,

        processor_path=MODEL_PATH

    )

    ########################################################
    # DataLoader
    ########################################################

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )

    ########################################################
    # Trainer
    ########################################################

    trainer = Trainer(

        encoder_name=ENCODER_NAME,

        model_path=MODEL_PATH,

        projection_dim=PROJECTION_DIM,

        freeze_encoder=FREEZE_ENCODER,

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        checkpoint_dir=CHECKPOINT_DIR

    )

    ########################################################
    # Model Information
    ########################################################

    trainable, total = count_parameters(trainer.model)

    print()

    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(validation_dataset)}")

    print()

    print("=" * 70)
    print("Model Information")
    print("=" * 70)

    print(f"Encoder           : {ENCODER_NAME}")
    print(f"Frozen Encoder    : {FREEZE_ENCODER}")
    print(f"Projection Dim    : {PROJECTION_DIM}")

    print()

    print(f"Trainable Params  : {trainable:,}")
    print(f"Total Params      : {total:,}")

    print("=" * 70)

    ########################################################
    # Start Training
    ########################################################

    history = trainer.fit(

        train_loader=train_loader,

        validation_loader=validation_loader,

        epochs=EPOCHS

    )

    ########################################################

    print()

    print("=" * 70)
    print("Training Finished Successfully")
    print("=" * 70)

    print()

    print("Best model saved to:")

    print(f"{CHECKPOINT_DIR}/best_model.pt")

    print()

    print("Last model saved to:")

    print(f"{CHECKPOINT_DIR}/last_model.pt")

    print()

    print("History saved to:")

    print(f"{CHECKPOINT_DIR}/history.csv")

    return history


############################################################

if __name__ == "__main__":

    main()