import os
import pandas as pd
import matplotlib.pyplot as plt


HISTORY = "checkpoints/history.csv"
OUTPUT_DIR = "results/train_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


history = pd.read_csv(HISTORY)
epochs = range(1, len(history) + 1)

plt.rcParams["figure.figsize"] = (8,5)

def plot(columns,labels,title,ylabel,filename):

    plt.figure()

    for c, l in zip(columns, labels):
        if c in history.columns:
            plt.plot(
                epochs,
                history[c],
                linewidth=2,
                label=l
            )

    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )
    plt.close()

# Loss
plot(
    ["train_loss","validation_loss"],
    ["Train","Validation"],
    "Training and Validation Loss",
    "Loss",
    "loss.png"
)

plot(
    [
        "gender_loss",
        "age_loss",
        "race_loss"
    ],

    [
        "Gender",
        "Age",
        "Race"
    ],

    "Task Loss",
    "Loss",
    "task_loss.png"
)

# Accuracy
plot(
    [
        "gender_accuracy",
        "age_accuracy",
        "race_accuracy"
    ],

    [
        "Gender",
        "Age",
        "Race"
    ],

    "Accuracy",
    "Accuracy (%)",
    "accuracy.png"
)

# Learning Rate
plot(
    [
        "learning_rate"
    ],

    [
        "Learning Rate"
    ],

    "Learning Rate",
    "LR",
    "learning_rate.png"
)

print("="*60)
print("Plots saved to:")
print(OUTPUT_DIR)
print("="*60)