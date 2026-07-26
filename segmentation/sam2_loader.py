import numpy as np
from transformers import pipeline
from face_crop import (
    crop_face_from_mask,
    select_face_mask
)

generator = pipeline(
    task="mask-generation",
    model="../checkpoints/sam2.1-hiera-large",
    device="cuda"
)

print("SAM2 Loaded Successfully")

def extract_face(image):

    image_np = np.array(image.convert("RGB"))

    masks = generator(image,points_per_batch=64)

    if len(masks["masks"]) == 0:
        return None

    best_mask = select_face_mask(
        image_np,
        masks
    )

    if best_mask is None:
        return None

    face = crop_face_from_mask(
        image_np,
        best_mask.cpu().numpy()
    )

    return face