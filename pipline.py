from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "train-*.parquet"
    }
)

sample = dataset["train"][2]

print(sample.keys())
print(sample["age"])
print(sample["gender"])
print(sample["race"])
sample["image"].show()