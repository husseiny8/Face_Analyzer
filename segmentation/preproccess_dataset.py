from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
from main import load_fairface
from sam2_loader import extract_face

# Output Directories
OUTPUT_DIR = Path("processed_faces")
LABEL_DIR = Path("labels")
LOG_DIR = Path("logs")

OUTPUT_DIR.mkdir(exist_ok=True)
LABEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Process One Dataset Split
def process_split(dataset, split_name):

    save_dir = OUTPUT_DIR / split_name
    save_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    logs = []

    print(f"\nProcessing {split_name} dataset ...")

    for index, sample in enumerate(tqdm(dataset, desc=split_name)):

        start_time = time.time()

        try:
            # Face Extraction
            face = extract_face(sample["image"])

            elapsed = round((time.time() - start_time) * 1000,2)

            # No Face Detected
            if face is None:

                logs.append({
                    "dataset": split_name,
                    "index": index,
                    "filename": "",
                    "status": "failed",
                    "gender": sample["gender"],
                    "age": sample["age"],
                    "race": sample["race"],
                    "time_ms": elapsed
                })

                continue

            # Save Face
            filename = f"{index:06d}.png"

            face.save(save_dir / filename)

            # Save Label
            labels.append({
                "filename": filename,
                "gender": sample["gender"],
                "age": sample["age"],
                "race": sample["race"]
            })

            # Save Log
            logs.append({
                "dataset": split_name,
                "index": index,
                "filename": filename,
                "status": "success",
                "gender": sample["gender"],
                "age": sample["age"],
                "race": sample["race"],
                "time_ms": elapsed
            })

        # Unexpected Error
        except Exception as e:
            elapsed = round((time.time() - start_time) * 1000,2)

            logs.append({
                "dataset": split_name,
                "index": index,
                "filename": "",
                "status": "error",
                "gender": sample["gender"],
                "age": sample["age"],
                "race": sample["race"],
                "time_ms": elapsed,
                "message": str(e)
            })

    # Save Labels
    labels_df = pd.DataFrame(labels)

    labels_df.to_csv(
        LABEL_DIR / f"{split_name}.csv",
        index=False
    )

    # Save Logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(
        LOG_DIR / f"{split_name}_log.csv",
        index=False
    )

    # Statistic
    total = len(logs)

    success = len(
        logs_df[
            logs_df["status"] == "success"
        ]
    )

    failed = len(
        logs_df[
            logs_df["status"] == "failed"
        ]
    )

    errors = len(
        logs_df[
            logs_df["status"] == "error"
        ]
    )

    success_rate = 0

    if total > 0:
        success_rate = success / total * 100

    print("\n" + "=" * 60)
    print(f"Split         : {split_name}")
    print(f"Total Images  : {total}")
    print(f"Saved Images  : {success}")
    print(f"No Face Found : {failed}")
    print(f"Errors        : {errors}")
    print(f"Success Rate  : {success_rate:.2f}%")
    print("=" * 60)

############################################################
# Main
############################################################

# if __name__ == "__main__":
def main():
    train_data, test_data, validation_data = load_fairface()

    ########################################################
    # فقط 5 تصویر اول Train
    ########################################################

    train_data = train_data.select(range(5))
    validation_data = validation_data.select(range(5))
    test_data = test_data.select(range(5))

    # Train
    process_split(train_data,"train")

    # Validation
    process_split(validation_data,"validation")

    # Test
    process_split(test_data,"test")

    print("\nDataset preprocessing completed successfully.")