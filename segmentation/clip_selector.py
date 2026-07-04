from PIL import Image
from transformers import CLIPProcessor
from transformers import CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"E:\FaceDetection\models\clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

model.eval()


def crop_from_mask(image, mask):
    # put zero in background
    output = image.copy()
    output[~mask] = 0

    return Image.fromarray(output)

def similarity(image):
    inputs = processor(
        # we can use only one prompt then in return should be like this:
        # return score.item()
        text=[
            "a human face",
            "face",
            "a person's face",
            "portrait",
            "human head"
        ],
        images=image,
        return_tensors="pt",
        padding=True
    )
    # added to use GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    score = outputs.logits_per_image

    return score.max().item()


def select_face_mask(image, masks):

    best_score = -1000

    best_mask = None

    for mask in masks:

        crop = crop_from_mask(
            image,
            mask["segmentation"]
        )

        score = similarity(crop)

        if score > best_score:

            best_score = score

            best_mask = mask

    return best_mask