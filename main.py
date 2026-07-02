from datasets import load_dataset

# 87000
Datas = load_dataset(
    "parquet",
    data_files={
        "train": "train-*.parquet"
    }
)

# 70000
train_data = Datas['train'][0:70000]
# 17000
test_data = Datas['train'][70000:]
# 11000
validation_data = load_dataset(
    "parquet",
    data_files={
        "train": "validation-*.parquet"
    }
)

# # name of column (image,gender,age,race)
# print(traindata["train"].column_names)
#
# # on sample of FairFace ds
# sample = traindata["train"][0]
#
# # show one of picture in ds
# import matplotlib.pyplot as plt
# plt.imshow(sample["image"])
# plt.show()
# # sample["image"].show()
