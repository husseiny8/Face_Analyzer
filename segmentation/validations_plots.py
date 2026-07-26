import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Recall: از بین نمونه‌های واقعی آن کلاس، چند درصد توسط مدل شناسایی شده‌اند؟

INPUT_DIR = "validation_results"
OUTPUT_DIR = "results/validation_plots"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Labels
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

plt.rcParams["figure.figsize"] = (8,6)


# Heatmap
def plot_confusion_matrix(csv_file,
                          labels,
                          title,
                          filename):

    cm = pd.read_csv(
        csv_file,
        index_col=0
    )

    plt.figure()

    plt.imshow(
        cm.values,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=45
    )

    plt.yticks(
        np.arange(len(labels)),
        labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm.iloc[i,j]),
                ha="center",
                va="center",
                fontsize=8,
                color="black"
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

# Confusion Matrices
plot_confusion_matrix(
    os.path.join(INPUT_DIR,
                 "gender_confusion_matrix.csv"),
    GENDER_LABELS,
    "Gender Confusion Matrix",
    "gender_confusion_matrix.png"
)

plot_confusion_matrix(
    os.path.join(INPUT_DIR,
                 "age_confusion_matrix.csv"),
    AGE_LABELS,
    "Age Confusion Matrix",
    "age_confusion_matrix.png"
)

plot_confusion_matrix(
    os.path.join(INPUT_DIR,
                 "race_confusion_matrix.csv"),
    RACE_LABELS,
    "Race Confusion Matrix",
    "race_confusion_matrix.png"
)

# Accuracy
accuracy = [
    94.83,
    60.02,
    72.33
]

tasks = [
    "Gender",
    "Age",
    "Race"
]

plt.figure()

plt.bar(
    tasks,
    accuracy
)

plt.ylim(0,100)

for i,v in enumerate(accuracy):

    plt.text(
        i,
        v+1,
        f"{v:.2f}%",
        ha="center"
    )

plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "accuracy.png"
    ),
    dpi=300
)

plt.close()

# Gender Metrics
gender_precision = [94.97,94.67]
gender_recall    = [95.27,94.34]
gender_f1        = [95.12,94.51]

x = np.arange(len(GENDER_LABELS))
w = 0.25

plt.figure()

plt.bar(
    x-w,
    gender_precision,
    width=w,
    label="Precision"
)

plt.bar(
    x,
    gender_recall,
    width=w,
    label="Recall"
)

plt.bar(
    x+w,
    gender_f1,
    width=w,
    label="F1"
)

plt.xticks(
    x,
    GENDER_LABELS
)

plt.ylabel("%")
plt.title("Gender Metrics")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "gender_metrics.png"
    ),
    dpi=300
)

plt.close()

# Age Precision / Recall
age_precision = [
75.36,78.98,58.98,64.30,50.43,
49.41,51.08,51.58,54.55
]

age_recall = [
78.39,84.51,40.05,72.48,52.70,
43.09,47.74,50.78,45.76
]

plt.figure()

plt.plot(
    AGE_LABELS,
    age_precision,
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    AGE_LABELS,
    age_recall,
    marker="s",
    linewidth=2,
    label="Recall"
)

plt.ylabel("%")
plt.title("Age Classification")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "age_metrics.png"
    ),
    dpi=300
)

plt.close()

# Race Precision / Recall
race_precision = [
72.36,81.23,89.39,
78.05,64.46,
56.35,63.29
]

race_recall = [
78.71,72.23,87.15,
76.07,67.66,
57.67,64.45
]

plt.figure()

plt.plot(
    RACE_LABELS,
    race_precision,
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    RACE_LABELS,
    race_recall,
    marker="s",
    linewidth=2,
    label="Recall"
)

plt.ylabel("%")
plt.title("Race Classification")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "race_metrics.png"
    ),
    dpi=300
)

plt.close()


print("="*60)
print("Plots saved to:")
print(OUTPUT_DIR)
print("="*60)

