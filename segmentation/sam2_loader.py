"""
Load SAM2, segment a photo, and keep only the mask that contains the face.

This is the "clip" of the project that turns a photo into a
background-removed face crop:

    1. Run SAM2's automatic mask generator on the image -> many candidate masks.
    2. Zero out everything outside that mask and save the result.

Run it as:
    python sam2_loader.py
    python sam2_loader.py --image /path/to/other/photo.jpg
    python sam2_loader.py --device cuda   # if you have a GPU available
"""

import matplotlib.pyplot as plt
from face_crop import *
import sys
from transformers import pipeline
from pathlib import Path
from CLIP_Head import main
from main import load_fairface

# main.py may sit next to this file, or one folder up (project root) -
# add both to sys.path so `from main import load_fairface` works either way.
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


generator = pipeline("mask-generation", "../checkpoints/sam2.1-hiera-large", device="cuda")

print("SAM2 Loaded Successfully")


# which FairFace sample to run: split is "train", "test", or "validation"
DATASET_SPLIT = "train"

# pull the image from the dataset loaded in main.py (instead of a hardcoded file)
train_data, test_data, validation_data = load_fairface()
dataset_splits = {"train": train_data, "test": test_data, "validation": validation_data}
print("Dataset Split Successfully")

for i in range(100,130):

    sample = dataset_splits[DATASET_SPLIT][i]
    # prepare image
    image = np.array(sample["image"].convert("RGB"))

        GENDER_MAP = {
            0: "Male",
            1: "Female"
        }

        gender = GENDER_MAP[sample["gender"]]


        AGE_MAP = {
            0: "0-2",
            1: "3-9",
            2: "10-19",
            3: "20-29",
            4: "30-39",
            5: "40-49",
            6: "50-59",
            7: "60-69",
            8: "70+"
        }


    if sample['age'] == 0:
        age = "0-2"
    elif sample['age'] == 1:
        age = "3-9"
    elif sample['age'] == 2:
        age = "10-19"
    elif sample['age'] == 3:
        age = "20-29"
    elif sample['age'] == 4:
        age = "30-39"
    elif sample['age'] == 5:
        age = "40-49"
    elif sample['age'] == 6:
        age = "50-59"
    elif sample['age'] == 7:
        age = "60-69"
    else:
        age = "+70"

        RACE_MAP = {
            0: "East Asian",
            1: "Indian",
            2: "Black",
            3: "White",
            4: "Middle Eastern",
            5: "Latino_Hispanic",
            6: "Southeast Asian"
        }

        race = RACE_MAP[sample["race"]]

    print(
        f"Loaded sample #{i} from the '{DATASET_SPLIT}' split "
        f"(age={age}, gender={gender}, race={race})"
    )

    # create a mask generator and pass the image to it so
    # after that we have all masks of our test image
    masks = generator(sample["image"], points_per_batch=64)
    print(f"{len(masks['masks'])} masks generated")
    # now we have all masks from that photo

    if len(masks['masks']) == 0:
        print("Error: No masks generated!")
        continue

    # show first image
    plt.imshow(image)
    plt.show()

    # # show all masks
    # for j, mask in enumerate(masks["masks"]):
    #     segmented = np.ones_like(image) * 255  # white background
    #     segmented[mask] = image[mask]
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(segmented)
    #     plt.title(f"Mask {j}")
    #     plt.axis("off")
    #     plt.show()

    best_mask = select_face_mask(image, masks)
    # show best mask (only face)
    face = crop_face_from_mask(
        image,
        best_mask.cpu().numpy()
    )

    plt.imshow(face)
    plt.axis("off")
    plt.show()
    main(face)

    # show the selected mask vs first image
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.title("Original")
    # plt.subplot(122)
    # plt.imshow(output)
    # plt.title(f"Masked ({DATASET_SPLIT}[{i}])")
    # plt.show()
    #
    # face.save(
    #     f"masks/masked_face_{DATASET_SPLIT}_{i}.png"
    # )
