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


def filter_masks(
    masks,
    image_shape,
    min_area_ratio=0.003,
    max_aspect_ratio=2.5,
    border_margin=10,
):

    H, W = image_shape[:2]
    image_area = H * W
    min_area = image_area * min_area_ratio
    min_width = int(W * 0.05)
    min_height = int(H * 0.05)
    filtered = []

    for mask in masks:

        seg = mask["segmentation"]

        area = seg.sum()

        if area < min_area:
            continue

        ys, xs = np.where(seg)

        if len(xs) == 0:
            continue

        x1 = xs.min()
        x2 = xs.max()

        y1 = ys.min()
        y2 = ys.max()

        width = x2 - x1 + 1
        height = y2 - y1 + 1

        aspect = max(width / height, height / width)

        if aspect > max_aspect_ratio:
            continue

        if width < min_width:
            continue

        if height < min_height:
            continue

        touches_border = (
            x1 <= border_margin or
            y1 <= border_margin or
            x2 >= W - border_margin or
            y2 >= H - border_margin
        )

        if touches_border:
            continue

        filtered.append(mask)

    return filtered

# Load SAM2 to the project
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
config = "configs/sam2/sam2_hiera_t.yaml"
predictor = build_sam2(config,checkpoint,device=device)
print("SAM2 Loaded Successfully")

# import a test image
image = Image.open("examples/race_Middle_Eastern.jpg").convert("RGB")
image = np.array(image)
print("Image Converted")

# create a mask generator and pass the image to it so
# after that we have all masks of our test image
mask_generator = SAM2AutomaticMaskGenerator(predictor)
print("Mask Generator Created")
masks = mask_generator.generate(image)
print(f"{len(masks)} masks generated")
# now we have all masks from that photo


# # show all masks
# plt.imshow(image)
# for mask in masks:
#     plt.contour(mask["segmentation"])
#     plt.show()

# height, width = image.shape[:2]
# image_area = height * width
# min_area = image_area * 0.1
#
# filtered_masks = []
#
# for mask in masks:
#
#     area = mask["segmentation"].sum()
#
#     if area >= min_area:
#         filtered_masks.append(mask)


masks = filter_masks(
    masks,
    image.shape
)
print(f"{len(masks)} selected masks")

best_mask = select_face_mask(image, masks)
# Zero background
output = image.copy()
output[~best_mask["segmentation"]] = 0

# show the selected mask
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(image)
plt.title("Original")
plt.subplot(122)
plt.imshow(output)
plt.title("Masked")
plt.show()

Image.fromarray(output).save(
    "masks/masked_face.png"
)