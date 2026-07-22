from PIL import Image
import numpy as np
from transformers import CLIPProcessor
from transformers import CLIPModel
import torch

def crop_face_from_mask(
    image,
    mask,
    padding=0.15,
    background_color=(0, 0, 0)
):
    """
    Parameters
    ----------
    image : numpy.ndarray (H,W,3)

    mask : numpy.ndarray (H,W)
        Boolean face mask.

    padding : float
        Percentage of padding around the face.

    Returns
    -------
    PIL.Image
        Cropped face with background removed.
    """

    if mask.dtype != bool:
        mask = mask.astype(bool)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    width = x_max - x_min
    height = y_max - y_min

    pad_x = int(width * padding)
    pad_y = int(height * padding)

    x_min = max(0, x_min - pad_x)
    x_max = min(image.shape[1], x_max + pad_x)

    y_min = max(0, y_min - pad_y)
    y_max = min(image.shape[0], y_max + pad_y)

    cropped = image[y_min:y_max, x_min:x_max].copy()

    cropped_mask = mask[y_min:y_max, x_min:x_max]

    output = np.full_like(cropped, background_color)

    output[cropped_mask] = cropped[cropped_mask]

    return Image.fromarray(output)

# The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder.
# These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"..\models\clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

model.eval()


def crop_from_mask(image, mask):
    # put zero in background
    output = image.copy()
    output[~mask] = 0

    face = Image.fromarray(output)

    face = face.resize(
        (224, 224),
        Image.Resampling.BICUBIC
    )

    return face
def similarity(image):
    prompts = [
        "only a human face",
        "a close-up photograph of a human face",
        "a portrait of a person's face",
        "front face",
        "human face",
        "a face occupying most of the image"
    ]

    inputs = processor(
        # we can use only one prompt then in return should be like this:
        # return score.item()
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    # added to use GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits_per_image
    probs = logits.softmax(dim=-1)

    return probs.max().item()

    # return score.item()


def select_face_mask(image, masks):

    best_score = -float("inf")
    best_mask = None

    for mask in masks['masks']:
        cropped = crop_face_from_mask(image,mask.cpu().numpy())
        # plt.imshow(cropped)
        # plt.show()
        score = similarity(cropped)

        if score > best_score:
            best_score = score
            best_mask = mask

    return best_mask