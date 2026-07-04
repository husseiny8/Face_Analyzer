from datasets import load_dataset
#
# dataset = load_dataset(
#     "parquet",
#     data_files={
#         "train": "train-*.parquet"
#     }
# )
#
# sample = dataset["train"][2]
#
# print(sample.keys())
# print(sample["age"])
# print(sample["gender"])
# print(sample["race"])
# sample["image"].show()

import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda")

x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

z = x @ y

print(z.device)