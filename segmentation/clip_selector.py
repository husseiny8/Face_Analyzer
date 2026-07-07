from PIL import Image
from transformers import CLIPProcessor
from transformers import CLIPModel
import torch

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

    return Image.fromarray(output)

def similarity(image):
    prompts = [
        "only a human face",
        "a close-up photograph of a human face",
        "a portrait of a person's face",
        "a selfie",
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
        cropped = crop_from_mask(image, mask.cpu().numpy())
        # plt.imshow(cropped)
        # plt.show()
        score = similarity(cropped)

        if score > best_score:
            best_score = score
            best_mask = mask

    return best_mask