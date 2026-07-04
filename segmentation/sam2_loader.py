from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Load SAM2 to th project
checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
config = "configs/sam2/sam2_hiera_t.yaml"
predictor = build_sam2(config,checkpoint,device="cpu")
print("SAM2 Loaded Successfully")

# import a test image
image = Image.open("race_Asian.jpg").convert("RGB")
image = np.array(image)

# create a mask generator and pass the image to it so
# after that we have all masks of our test image
mask_generator = SAM2AutomaticMaskGenerator(predictor)
masks = mask_generator.generate(image)
# now we have all masks from that photo


# # show all masks
# plt.imshow(image)
# for mask in masks:
#     plt.contour(mask["segmentation"])
#     plt.show()


# find the largest mask
largest = max(masks,key = lambda x: x["area"])
mask = largest["segmentation"]

# Zero background
output = image.copy()
output[~mask] = 0

# show selected mask(largest)
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(image)
plt.title("Original")
plt.subplot(122)
plt.imshow(output)
plt.title("Masked")
plt.show()

Image.fromarray(output).save(
    "masked_face.png"
)