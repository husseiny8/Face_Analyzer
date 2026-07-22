from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = r"..\models\clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

model.eval()


# -----------------------------
# Generic CLIP classifier
# -----------------------------
def classify(image, prompts):
    """
    Classify an image using CLIP and a list of text prompts.

    Returns:
        {
            "label": predicted label,
            "index": predicted class index,
            "confidence": confidence score,
            "probabilities": list of probabilities
        }
    """

    inputs = processor(
        images=image,
        text=prompts,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = outputs.logits_per_image.softmax(dim=-1)

    confidence, index = probabilities.max(dim=-1)

    return {
        "label": prompts[index.item()],
        # "index": index.item(),
        "confidence": f"{round(confidence.item() * 100)}%",
        # "probabilities": probabilities.squeeze(0).cpu().tolist()
    }


# -----------------------------
# Age Head
# -----------------------------
AGE_PROMPTS = [
    "a portrait of a person between 0 and 2 years old",
    "a portrait of a person between 3 and 9 years old",
    "a portrait of a person between 10 and 19 years old",
    "a portrait of a person between 20 and 29 years old",
    "a portrait of a person between 30 and 39 years old",
    "a portrait of a person between 40 and 49 years old",
    "a portrait of a person between 50 and 59 years old",
    "a portrait of a person between 60 and 69 years old",
    "a portrait of a person older than 70 years"
]


def age_head(image):
    return classify(image, AGE_PROMPTS)


# -----------------------------
# Gender Head
# -----------------------------
GENDER_PROMPTS = [
    "a portrait of a man",
    "a portrait of a woman"
]


def gender_head(image):
    return classify(image, GENDER_PROMPTS)


# -----------------------------
# Race Head (FairFace Classes)
# -----------------------------
RACE_PROMPTS = [
    "a portrait of an East Asian person",
    "a portrait of an Indian person",
    "a portrait of a Black person",
    "a portrait of a White person",
    "a portrait of a Middle Eastern person",
    "a portrait of a Latino or Hispanic person",
    "a portrait of a Southeast Asian person"
]


def race_head(image):
    return classify(image, RACE_PROMPTS)


def main(image):
    age = age_head(image)
    gender = gender_head(image)
    race = race_head(image)
    print(age)
    print(gender)
    print(race)
    print("============================================")

