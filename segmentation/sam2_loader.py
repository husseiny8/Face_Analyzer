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

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from clip_selector import select_face_mask
import sys
from transformers import pipeline
from pathlib import Path

# main.py may sit next to this file, or one folder up (project root) -
# add both to sys.path so `from main import load_fairface` works either way.
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR, _THIS_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main import load_fairface


# Load SAM2 to the project
# device = "cuda" if torch.cuda.is_available() else "cpu"
# checkpoint = "../checkpoints/sam2_hiera_large.pt"
# config = "../sam2/configs/sam2/sam2_hiera_l.yaml"
# predictor = build_sam2(config,checkpoint,device=device)
generator = pipeline("mask-generation", "../checkpoints/sam2.1-hiera-large", device="cuda")

print("SAM2 Loaded Successfully")


# which FairFace sample to run: split is "train", "test", or "validation"
DATASET_SPLIT = "train"

# pull the image from the dataset loaded in main.py (instead of a hardcoded file)
train_data, test_data, validation_data = load_fairface()
dataset_splits = {"train": train_data, "test": test_data, "validation": validation_data}
print("Dataset Split Successfully")

for i in range(10,30):

    sample = dataset_splits[DATASET_SPLIT][i]
    # prepare image
    image = np.array(sample["image"].convert("RGB"))
    print(
        f"Loaded sample #{i} from the '{DATASET_SPLIT}' split "
        f"(age={sample['age']}, gender={sample['gender']}, race={sample['race']})"
    )

    # create a mask generator and pass the image to it so
    # after that we have all masks of our test image
    # mask_generator = SAM2AutomaticMaskGenerator(predictor)
    # print("Mask Generator Created")
    # masks = mask_generator.generate(image)
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
    segmented = np.ones_like(image) * 255  # white background
    segmented[best_mask] = image[best_mask]
    plt.figure(figsize=(6, 6))
    plt.imshow(segmented)
    plt.title("Mask")
    plt.axis("off")
    plt.show()


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
    # Image.fromarray(output).save(
    #     f"masks/masked_face_{DATASET_SPLIT}_{i}.png"
    # )
