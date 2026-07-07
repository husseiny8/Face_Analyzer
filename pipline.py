import numpy as np
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
from segmentation.clip_selector import select_face_mask

generator = pipeline("mask-generation", "checkpoints/sam2.1-hiera-large", device="cuda")

# <class 'PIL.JpegImagePlugin.JpegImageFile'>
sample = Image.open("images (3).jpg")
image = np.array(sample.convert("RGB"))

masks = generator(sample, points_per_batch=64)

print(len(masks['masks']))
print(masks.keys())

# show all masks
plt.imshow(image)
plt.show()
#
# for i in range(len(masks['masks'])):
#     plt.imshow((masks['masks'][i]))
#     plt.show()
for i, mask in enumerate(masks["masks"]):

    segmented = np.ones_like(image) * 255  # white background
    segmented[mask] = image[mask]

    plt.figure(figsize=(6,6))
    plt.imshow(segmented)
    plt.title(f"Mask {i}")
    plt.axis("off")
    plt.show()