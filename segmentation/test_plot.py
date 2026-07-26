import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "test_results"

PREDICTIONS_FILE = "test_result/predictions.csv"
SUMMARY_FILE = "test_result/test_summary.csv"
GENDER_CM_FILE = "test_result/confusion_matrix_gender.csv"
AGE_CM_FILE = "test_result/confusion_matrix_age.csv"
RACE_CM_FILE = "test_result/confusion_matrix_race.csv"
OUTPUT_DIR = "results/test_plots"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

############################################################
# Load Files
############################################################

predictions = pd.read_csv(PREDICTIONS_FILE)

summary = pd.read_csv(SUMMARY_FILE)

gender_cm = pd.read_csv(
    GENDER_CM_FILE,
    index_col=0
)

age_cm = pd.read_csv(
    AGE_CM_FILE,
    index_col=0
)

race_cm = pd.read_csv(
    RACE_CM_FILE,
    index_col=0
)

############################################################
# Labels
############################################################

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
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern"
]

plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams["font.size"] = 11

############################################################
# Accuracy Plot
############################################################

def plot_accuracy():

    plt.figure()

    values = summary["Accuracy"] * 100

    plt.bar(
        summary["Task"],
        values,
        width=0.5
    )

    plt.ylim(0,100)

    for i,v in enumerate(values):

        plt.text(
            i,
            v + 1,
            f"{v:.2f}%",
            ha="center",
            fontsize=11
        )

    plt.grid(axis="y")

    plt.xlabel("Task")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "accuracy.png"
        ),
        dpi=300
    )

    plt.close()

############################################################
# Confusion Matrix
############################################################

def plot_confusion_matrix(
        cm,
        title,
        filename
):

    fig, ax = plt.subplots(
        figsize=(8,7)
    )

    image = ax.imshow(
        cm.values,
        cmap="Blues"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        np.arange(cm.shape[1])
    )

    ax.set_yticks(
        np.arange(cm.shape[0])
    )

    ax.set_xticklabels(
        cm.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        cm.index
    )

    ax.set_xlabel(
        "Predicted Class",
        fontsize=12
    )

    ax.set_ylabel(
        "True Class",
        fontsize=12
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    threshold = cm.values.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            value = int(cm.iloc[i,j])

            color = "white"

            if value < threshold:
                color = "black"

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                fontsize=9,
                color=color
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

############################################################
# Per-Class Accuracy
############################################################

def compute_per_class_accuracy(cm):

    accuracy = []

    for i in range(cm.shape[0]):

        correct = cm.iloc[i,i]

        total = cm.iloc[i].sum()

        accuracy.append(
            100 * correct / total
        )

    return accuracy


def plot_per_class_accuracy(
        cm,
        labels,
        title,
        filename
):

    acc = compute_per_class_accuracy(cm)

    plt.figure(figsize=(10,5))

    bars = plt.bar(
        labels,
        acc
    )

    plt.ylim(0,100)

    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(title)

    plt.grid(axis="y")

    for bar, value in zip(bars, acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


############################################################
# Plot All
############################################################

plot_accuracy()

plot_confusion_matrix(
    gender_cm,
    "Gender Confusion Matrix",
    "gender_confusion_matrix.png"
)

plot_confusion_matrix(
    age_cm,
    "Age Confusion Matrix",
    "age_confusion_matrix.png"
)

plot_confusion_matrix(
    race_cm,
    "Race Confusion Matrix",
    "race_confusion_matrix.png"
)

plot_per_class_accuracy(
    age_cm,
    AGE_LABELS,
    "Age Per-Class Accuracy",
    "age_per_class_accuracy.png"
)

plot_per_class_accuracy(
    race_cm,
    RACE_LABELS,
    "Race Per-Class Accuracy",
    "race_per_class_accuracy.png"
)

############################################################
# Print Summary
############################################################

print()
print("=" * 60)
print("Plots Saved Successfully")
print("=" * 60)
print()

print("Output Directory :")
print(OUTPUT_DIR)

print()

print("Generated Files")

print()

print("accuracy.png")
print("gender_confusion_matrix.png")
print("age_confusion_matrix.png")
print("race_confusion_matrix.png")
print("age_per_class_accuracy.png")
print("race_per_class_accuracy.png")

print()
print("=" * 60)