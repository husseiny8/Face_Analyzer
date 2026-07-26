"""
test.py

Evaluate trained Multi-Task model on the Test set.

Outputs
-------
- Overall Accuracy
- Gender Accuracy
- Age Accuracy
- Race Accuracy

- Confusion Matrix
- Classification Report

- prediction.csv
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import pandas as pd
from tqdm import tqdm

from dataset import FaceDataset
from main import load_fairface
from multitask_model import MultiTaskModel

def main():

    ############################################################
    # Device
    ############################################################

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print("=" * 60)
    print("Testing")
    print("=" * 60)
    print(f"Device : {device}")

    ############################################################
    # Dataset Selection
    ############################################################

    print()
    print("=" * 60)
    print("Dataset Selection")
    print("=" * 60)
    print("1) Original FairFace Images")
    print("2) SAM2 Cropped Faces")
    print("=" * 60)

    choice = input("Select (1 or 2): ").strip()

    ############################################################
    # Original FairFace
    ############################################################

    if choice == "1":

        _, test_data, _ = load_fairface()

        test_dataset = FaceDataset(

            dataset=test_data,

            encoder_name="clip"

        )

        dataset_mode = "Original FairFace"

    ############################################################
    # SAM Dataset
    ############################################################

    else:

        test_dataset = FaceDataset(

            image_dir="processed_faces/test",

            csv_file="labels/test.csv",

            encoder_name="clip"

        )

        dataset_mode = "SAM2 Cropped Faces"

    ############################################################
    # DataLoader
    ############################################################

    test_loader = DataLoader(

        test_dataset,

        batch_size=32,

        shuffle=False,

        num_workers=4,

        pin_memory=True

    )

    print()
    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"Dataset Mode : {dataset_mode}")
    print(f"Test Images  : {len(test_dataset)}")

    ############################################################
    # Load Model
    ############################################################

    model = MultiTaskModel(

        encoder_name="clip",

        freeze_encoder=True

    )

    checkpoint = torch.load(

        "checkpoints/best_model.pt",

        map_location=device

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.to(device)

    model.eval()

    print()
    print("=" * 60)
    print("Best Model Loaded Successfully")
    print("=" * 60)

    ############################################################
    # Containers
    ############################################################

    gender_true = []
    gender_pred = []

    age_true = []
    age_pred = []

    race_true = []
    race_pred = []

    prediction_rows = []

    ############################################################
    # Inference
    ############################################################

    print()
    print("=" * 60)
    print("Running Inference...")
    print("=" * 60)

    with torch.no_grad():

        progress = tqdm(

            test_loader,

            desc="Testing"

        )

        image_index = 0

        for batch in progress:

            ####################################################
            # Move To Device
            ####################################################

            images = batch["pixel_values"].to(device)

            gender_labels = batch["gender"].cpu()

            age_labels = batch["age"].cpu()

            race_labels = batch["race"].cpu()

            ####################################################
            # Prediction
            ####################################################

            outputs = model.predict(images)

            ####################################################
            # Convert To CPU
            ####################################################

            gender_prediction = outputs[
                "gender_prediction"
            ].cpu().view(-1)

            age_prediction = outputs[
                "age_prediction"
            ].cpu()

            race_prediction = outputs[
                "race_prediction"
            ].cpu()

            ####################################################
            # Save Labels
            ####################################################

            gender_true.extend(
                gender_labels.numpy().tolist()
            )

            gender_pred.extend(
                gender_prediction.numpy().tolist()
            )

            age_true.extend(
                age_labels.numpy().tolist()
            )

            age_pred.extend(
                age_prediction.numpy().tolist()
            )

            race_true.extend(
                race_labels.numpy().tolist()
            )

            race_pred.extend(
                race_prediction.numpy().tolist()
            )

            ####################################################
            # Save Prediction CSV
            ####################################################

            batch_size = len(age_labels)

            gender_prob = outputs[
                "gender_probability"
            ].cpu().view(-1)

            age_prob = outputs[
                "age_probability"
            ].cpu()

            race_prob = outputs[
                "race_probability"
            ].cpu()

            for i in range(batch_size):
                prediction_rows.append(

                    {

                        "index": image_index,

                        "gender_true": int(gender_labels[i]),

                        "gender_pred": int(gender_prediction[i]),

                        "gender_probability":
                            float(gender_prob[i]),

                        "age_true": int(age_labels[i]),

                        "age_pred": int(age_prediction[i]),

                        "age_confidence":
                            float(age_prob[i].max()),

                        "race_true": int(race_labels[i]),

                        "race_pred": int(race_prediction[i]),

                        "race_confidence":
                            float(race_prob[i].max())

                    }

                )

                image_index += 1

    print()
    print("=" * 60)
    print("Inference Finished")
    print("=" * 60)

    ############################################################
    # Evaluation
    ############################################################

    gender_accuracy = accuracy_score(
        gender_true,
        gender_pred
    )

    age_accuracy = accuracy_score(
        age_true,
        age_pred
    )

    race_accuracy = accuracy_score(
        race_true,
        race_pred
    )

    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)

    print(f"Gender Accuracy : {gender_accuracy * 100:.2f}%")
    print(f"Age Accuracy    : {age_accuracy * 100:.2f}%")
    print(f"Race Accuracy   : {race_accuracy * 100:.2f}%")

    ############################################################
    # Classification Report
    ############################################################

    print()
    print("=" * 60)
    print("Gender Classification Report")
    print("=" * 60)

    print(

        classification_report(

            gender_true,

            gender_pred,

            digits=4

        )

    )

    print()
    print("=" * 60)
    print("Age Classification Report")
    print("=" * 60)

    print(

        classification_report(

            age_true,

            age_pred,

            digits=4

        )

    )

    print()
    print("=" * 60)
    print("Race Classification Report")
    print("=" * 60)

    print(

        classification_report(

            race_true,

            race_pred,

            digits=4

        )

    )

    ############################################################
    # Confusion Matrix
    ############################################################

    gender_cm = confusion_matrix(

        gender_true,

        gender_pred

    )

    age_cm = confusion_matrix(

        age_true,

        age_pred

    )

    race_cm = confusion_matrix(

        race_true,

        race_pred

    )

    ############################################################
    # Save Predictions
    ############################################################

    prediction_df = pd.DataFrame(

        prediction_rows

    )

    prediction_df.to_csv(

        "predictions.csv",

        index=False

    )

    ############################################################
    # Save Confusion Matrices
    ############################################################

    pd.DataFrame(

        gender_cm

    ).to_csv(

        "confusion_matrix_gender.csv",

        index=False

    )

    pd.DataFrame(

        age_cm

    ).to_csv(

        "confusion_matrix_age.csv",

        index=False

    )

    pd.DataFrame(

        race_cm

    ).to_csv(

        "confusion_matrix_race.csv",

        index=False

    )

    ############################################################
    # Save Summary
    ############################################################

    summary = pd.DataFrame({

        "Task": [

            "Gender",

            "Age",

            "Race"

        ],

        "Accuracy": [

            gender_accuracy,

            age_accuracy,

            race_accuracy

        ]

    })

    summary.to_csv(

        "test_summary.csv",

        index=False

    )

    ############################################################
    # Finish
    ############################################################

    print()
    print("=" * 60)
    print("Evaluation Finished Successfully")
    print("=" * 60)

    print()

    print("Files Generated:")

    print("  predictions.csv")

    print("  test_summary.csv")

    print("  confusion_matrix_gender.csv")

    print("  confusion_matrix_age.csv")

    print("  confusion_matrix_race.csv")

    print()

    print("Done.")

if __name__ == "__main__":
    main()