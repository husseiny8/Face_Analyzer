from pathlib import Path
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_fairface():

    Datas = load_dataset(
        "parquet",
        data_files={
            "train": str(DATA_DIR / "train-*.parquet")
        }
    )

    train_data = Datas["train"].select(range(0, 70000))

    test_data = Datas["train"].select(
        range(70000, len(Datas["train"]))
    )

    validation_data = load_dataset(
        "parquet",
        data_files={
            "validation": str(DATA_DIR / "validation-*.parquet")
        }
    )["validation"]

    return train_data, test_data, validation_data


if __name__ == "__main__":
    train_data, test_data, validation_data = load_fairface()

    print(type(train_data['image'][0]))
    print(type(validation_data['image'][0]))
    print(type(test_data['image'][0]))

    # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    # <class 'PIL.JpegImagePlugin.JpegImageFile'>

# # show one of picture in dataset
# import matplotlib.pyplot as plt
# plt.imshow(sample["image"])
# plt.show()
# # sample["image"].show()
