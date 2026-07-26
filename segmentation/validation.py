import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor
from multitask_model import MultiTaskModel

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Validation")
print("=" * 60)
print(f"Device : {device}")

CHECKPOINT = "checkpoints/best_model.pt"

MODEL_PATH = "../models/clip-vit-base-patch32"

PROCESSOR_PATH = "../models/clip-vit-base-patch32"

DATA_DIR = Path("../data")

OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLES_DIR = OUTPUT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

GENDER_LABELS = [
    "Male",
    "Female"
]

AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]

RACE_LABELS = [
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian"
]


processor = CLIPImageProcessor.from_pretrained(
    PROCESSOR_PATH
)

validation_data = load_dataset(
    "parquet",
    data_files={
        "validation": str(DATA_DIR / "validation-*.parquet")
    }
)["validation"]


# validation_data = (
#     validation_data
#     .shuffle(seed=42)
#     .select(range(1000))
# )

print()
print("=" * 60)
print("Dataset Information")
print("=" * 60)
print(f"Validation Images : {len(validation_data)}")

model = MultiTaskModel(
    encoder_name="clip",
    model_path=MODEL_PATH,
    freeze_encoder=True,
    projection_dim=512
)

checkpoint = torch.load(
    CHECKPOINT,
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


def preprocess_image(image):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    return inputs["pixel_values"].to(device)


def probability(prob_tensor, index):
    return prob_tensor[0, index].item() * 100

# Validation
@torch.no_grad()
def run_validation():

    print()
    print("=" * 60)
    print("Running Validation...")
    print("=" * 60)

    results = []

    total_time = 0

    for index, sample in enumerate(tqdm(validation_data)):

        image = sample["image"]

        pixel_values = preprocess_image(image)

        start_time = time.time()

        outputs = model.predict(pixel_values)

        inference_time = (time.time() - start_time) * 1000

        total_time += inference_time

        gt_gender = sample["gender"]
        gt_age = sample["age"]
        gt_race = sample["race"]


        gender_pred = outputs["gender_prediction"].item()
        age_pred = outputs["age_prediction"].item()
        race_pred = outputs["race_prediction"].item()

        # Confidence
        gender_prob = outputs["gender_probability"].item()

        if gender_pred == 0:
            gender_confidence = (1.0 - gender_prob) * 100
        else:
            gender_confidence = gender_prob * 100

        age_confidence = probability(
            outputs["age_probability"],
            age_pred
        )

        race_confidence = probability(
            outputs["race_probability"],
            race_pred
        )


        # Save 10 sample images (random)
        import random
        sample_inc = random.sample(range(len(validation_data)),10)
        if index in sample_inc:
            plt.figure(figsize=(6, 7))

            plt.imshow(image)
            plt.axis("off")

            plt.title(
                f"Gender : {GENDER_LABELS[gt_gender]} → {GENDER_LABELS[gender_pred]} ({gender_confidence:.1f}%)\n"
                f"Age : {AGE_LABELS[gt_age]} → {AGE_LABELS[age_pred]} ({age_confidence:.1f}%)\n"
                f"Race : {RACE_LABELS[gt_race]} → {RACE_LABELS[race_pred]} ({race_confidence:.1f}%)",
                fontsize=11
            )

            plt.savefig(
                SAMPLES_DIR / f"sample_{index:02d}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        # Save Result
        results.append({
            "index": index,
            "gender_true": gt_gender,
            "age_true": gt_age,
            "race_true": gt_race,
            "gender_pred": gender_pred,
            "age_pred": age_pred,
            "race_pred": race_pred,
            "gender_label": GENDER_LABELS[gender_pred],
            "age_label": AGE_LABELS[age_pred],
            "race_label": RACE_LABELS[race_pred],
            "gender_probability": round(gender_confidence,2),
            "age_probability": round(age_confidence,2),
            "race_probability": round(race_confidence,2),
            "gender_correct": int(gt_gender == gender_pred),
            "age_correct": int(gt_age == age_pred),
            "race_correct": int(gt_race == race_pred),
            "time_ms": round(inference_time,2)
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False
    )

    print()

    print("=" * 60)
    print("Validation Finished")
    print("=" * 60)

    print()

    print("Prediction File Saved :")
    print(OUTPUT_DIR / "validation_predictions.csv")

    print()

    average_time = total_time / len(validation_data)

    print(f"Average Inference Time : {average_time:.2f} ms")

    return results_df

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate(results_df):
    gender_accuracy = accuracy_score(
        results_df["gender_true"],
        results_df["gender_pred"]
    )

    age_accuracy = accuracy_score(
        results_df["age_true"],
        results_df["age_pred"]
    )

    race_accuracy = accuracy_score(
        results_df["race_true"],
        results_df["race_pred"]
    )

    gender_report = classification_report(
        results_df["gender_true"],
        results_df["gender_pred"],
        digits=4
    )

    age_report = classification_report(
        results_df["age_true"],
        results_df["age_pred"],
        digits=4
    )

    race_report = classification_report(
        results_df["race_true"],
        results_df["race_pred"],
        digits=4
    )

    # Confusion Matrix
    gender_cm = confusion_matrix(
        results_df["gender_true"],
        results_df["gender_pred"]
    )

    age_cm = confusion_matrix(
        results_df["age_true"],
        results_df["age_pred"]
    )

    race_cm = confusion_matrix(
        results_df["race_true"],
        results_df["race_pred"]
    )

    # Save Confusion Matrices
    pd.DataFrame(
        gender_cm,
        index=GENDER_LABELS,
        columns=GENDER_LABELS
    ).to_csv(
        OUTPUT_DIR / "gender_confusion_matrix.csv"
    )

    pd.DataFrame(
        age_cm,
        index=AGE_LABELS,
        columns=AGE_LABELS
    ).to_csv(
        OUTPUT_DIR / "age_confusion_matrix.csv"
    )

    pd.DataFrame(
        race_cm,
        index=RACE_LABELS,
        columns=RACE_LABELS
    ).to_csv(
        OUTPUT_DIR / "race_confusion_matrix.csv"
    )

    # Save Reports
    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 60 + "\n")
        f.write("Gender Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(gender_report)

        f.write("\n\n")

        f.write("=" * 60 + "\n")
        f.write("Age Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(age_report)

        f.write("\n\n")

        f.write("=" * 60 + "\n")
        f.write("Race Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(race_report)

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    print()

    print(f"Gender Accuracy : {gender_accuracy*100:.2f}%")
    print(f"Age Accuracy    : {age_accuracy*100:.2f}%")
    print(f"Race Accuracy   : {race_accuracy*100:.2f}%")

    print()

    print("=" * 60)
    print("Files Saved")
    print("=" * 60)

    print()

    print("validation_predictions.csv")
    print("classification_report.txt")
    print("gender_confusion_matrix.csv")
    print("age_confusion_matrix.csv")
    print("race_confusion_matrix.csv")

    print()

def main():

    results_df = run_validation()

    evaluate(results_df)

    print()
    print("=" * 60)
    print("Validation Finished Successfully")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()