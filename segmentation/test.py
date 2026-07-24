import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from dataset import FaceDataset
from trainer import Trainer


############################################################
# Configuration
############################################################

TEST_IMAGE_DIR = "processed_faces/test"

TEST_CSV = "labels/test.csv"

CHECKPOINT = "checkpoints/best_model.pt"

BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################################################
# Dataset
############################################################

test_dataset = FaceDataset(

    image_dir=TEST_IMAGE_DIR,

    csv_file=TEST_CSV,

    encoder_name="clip"

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=4,

    pin_memory=True

)


############################################################
# Trainer
############################################################

trainer = Trainer(

    encoder_name="clip",

    freeze_encoder=True

)

trainer.load_checkpoint("best_model.pt")

model = trainer.model

model.eval()


############################################################
# Loss
############################################################

gender_loss_fn = nn.BCEWithLogitsLoss()

age_loss_fn = nn.CrossEntropyLoss()

race_loss_fn = nn.CrossEntropyLoss()


############################################################
# Statistics
############################################################

gender_true = []
gender_pred = []

age_true = []
age_pred = []

race_true = []
race_pred = []

total_loss = 0.0


############################################################
# Evaluation
############################################################

with torch.no_grad():

    for batch in test_loader:

        images = batch["pixel_values"].to(DEVICE)

        gender = batch["gender"].to(DEVICE).unsqueeze(1)

        age = batch["age"].to(DEVICE)

        race = batch["race"].to(DEVICE)

        outputs = model(images)

        ####################################################

        gender_loss = gender_loss_fn(

            outputs["gender"],

            gender

        )

        age_loss = age_loss_fn(

            outputs["age"],

            age

        )

        race_loss = race_loss_fn(

            outputs["race"],

            race

        )

        loss = (

            gender_loss +

            age_loss +

            race_loss

        )

        total_loss += loss.item()

        ####################################################
        # Predictions
        ####################################################

        gender_prediction = (

            torch.sigmoid(outputs["gender"]) >= 0.5

        ).long()

        age_prediction = outputs["age"].argmax(dim=1)

        race_prediction = outputs["race"].argmax(dim=1)

        ####################################################

        gender_true.extend(

            gender.cpu().numpy().flatten()

        )

        gender_pred.extend(

            gender_prediction.cpu().numpy().flatten()

        )

        ####################################################

        age_true.extend(

            age.cpu().numpy()

        )

        age_pred.extend(

            age_prediction.cpu().numpy()

        )

        ####################################################

        race_true.extend(

            race.cpu().numpy()

        )

        race_pred.extend(

            race_prediction.cpu().numpy()

        )


############################################################
# Metrics
############################################################

results = []

for name, gt, pred in [

    ("Gender", gender_true, gender_pred),

    ("Age", age_true, age_pred),

    ("Race", race_true, race_pred)

]:

    accuracy = accuracy_score(gt, pred)

    precision = precision_score(

        gt,

        pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        gt,

        pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        gt,

        pred,

        average="weighted",

        zero_division=0

    )

    results.append({

        "Task": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1

    })

    ########################################################
    # Confusion Matrix
    ########################################################

    cm = confusion_matrix(gt, pred)

    plt.figure(figsize=(6,6))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues"

    )

    plt.title(f"{name} Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.tight_layout()

    plt.savefig(

        f"checkpoints/{name.lower()}_confusion_matrix.png"

    )

    plt.close()


############################################################
# Save Metrics
############################################################

results = pd.DataFrame(results)

results.to_csv(

    "checkpoints/test_results.csv",

    index=False

)


############################################################
# Print
############################################################

print("=" * 60)

print("Test Results")

print("=" * 60)

print(results)

print()

print(

    f"Average Test Loss : "

    f"{total_loss/len(test_loader):.4f}"

)