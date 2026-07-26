import os
import time
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformers import CLIPImageProcessor

from multitask_model import MultiTaskModel


############################################################
# Device
############################################################

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print("=" * 60)
print("Prediction")
print("=" * 60)
print(f"Device : {device}")


############################################################
# Paths
############################################################

CHECKPOINT = "checkpoints/best_model.pt"

PROCESSOR_PATH = "../models/clip-vit-base-patch32"

MODEL_PATH = "../models/clip-vit-base-patch32"

RESULT_DIR = Path("results")

RESULT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


############################################################
# Labels
############################################################

GENDER_LABELS = [

    "Male",

    "Female"

]


AGE_LABELS = [

    "0-2",

    "3-9",

    "10-19",

    "20-29",

    "30-39",

    "40-49",

    "50-59",

    "60-69",

    "70+"

]


RACE_LABELS = [

    "White",

    "Black",

    "Latino_Hispanic",

    "East Asian",

    "Southeast Asian",

    "Indian",

    "Middle Eastern"

]


############################################################
# Load Processor
############################################################

processor = CLIPImageProcessor.from_pretrained(

    PROCESSOR_PATH

)


############################################################
# Load Model
############################################################

model = MultiTaskModel(

    encoder_name="clip",

    model_path=MODEL_PATH,

    freeze_encoder=True,

    projection_dim=512

)

checkpoint = torch.load(

    CHECKPOINT,

    map_location=device

)

model.load_state_dict(

    checkpoint["model_state_dict"]

)

model.to(device)

model.eval()

print()

print("=" * 60)
print("Best Model Loaded Successfully")
print("=" * 60)


############################################################
# Image Preprocessing
############################################################

def preprocess_image(image_path):

    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )

    inputs = processor(

        images=image,

        return_tensors="pt"

    )

    pixel_values = inputs["pixel_values"].to(

        device

    )

    return image, pixel_values


############################################################
# Softmax Probability
############################################################

def get_probability(probability_tensor, index):

    return probability_tensor[0, index].item() * 100

############################################################
# Predict One Image
############################################################

@torch.no_grad()
def predict_image(image_path):

    ########################################################
    # Load Image
    ########################################################

    image, pixel_values = preprocess_image(image_path)

    ########################################################
    # Inference
    ########################################################

    start_time = time.time()

    outputs = model.predict(pixel_values)

    inference_time = (time.time() - start_time) * 1000

    ########################################################
    # Gender
    ########################################################

    gender_index = outputs["gender_prediction"].item()

    gender_probability = (

        outputs["gender_probability"]

        .squeeze()

        .item()

    )

    if gender_index == 0:

        gender_confidence = gender_probability

    else:

        gender_confidence = 1.0 - gender_probability

    gender_confidence *= 100

    ########################################################
    # Age
    ########################################################

    age_index = outputs["age_prediction"].item()

    age_probability = get_probability(

        outputs["age_probability"],

        age_index

    )

    ########################################################
    # Race
    ########################################################

    race_index = outputs["race_prediction"].item()

    race_probability = get_probability(

        outputs["race_probability"],

        race_index

    )

    ########################################################
    # Print Result
    ########################################################

    print()

    print("=" * 60)

    print("Prediction Result")

    print("=" * 60)

    print()

    print(f"Image : {image_path}")

    print()

    print(

        f"Gender : {GENDER_LABELS[gender_index]}"

    )

    print(

        f"Probability : {gender_confidence:.2f}%"

    )

    print()

    print(

        f"Age : {AGE_LABELS[age_index]}"

    )

    print(

        f"Probability : {age_probability:.2f}%"

    )

    print()

    print(

        f"Race : {RACE_LABELS[race_index]}"

    )

    print(

        f"Probability : {race_probability:.2f}%"

    )

    print()

    print(

        f"Inference Time : {inference_time:.2f} ms"

    )

    print("=" * 60)

    ########################################################
    # Return Results
    ########################################################

    return {

        "image": image,

        "gender": GENDER_LABELS[gender_index],

        "gender_probability": gender_confidence,

        "age": AGE_LABELS[age_index],

        "age_probability": age_probability,

        "race": RACE_LABELS[race_index],

        "race_probability": race_probability,

        "time": inference_time

    }
############################################################
# Main
############################################################

def main():

    print()

    print("=" * 60)
    print("Single Image Prediction")
    print("=" * 60)

    image_path = input(

        "Enter image path : "

    ).strip()

    ########################################################

    if not os.path.exists(image_path):

        print()

        print("Image not found!")

        return

    ########################################################

    result = predict_image(

        image_path

    )

    ########################################################
    # Display Result
    ########################################################

    plt.figure(

        figsize=(8, 8)

    )

    plt.imshow(

        result["image"]

    )

    plt.axis("off")

    plt.title(

        f"Gender : {result['gender']} ({result['gender_probability']:.1f}%)\n"

        f"Age : {result['age']} ({result['age_probability']:.1f}%)\n"

        f"Race : {result['race']} ({result['race_probability']:.1f}%)",

        fontsize=12

    )

    ########################################################
    # Save Figure
    ########################################################

    output_path = RESULT_DIR / (

        Path(image_path).stem + "_prediction.png"

    )

    plt.savefig(

        output_path,

        dpi=200,

        bbox_inches="tight"

    )

    plt.show()

    ########################################################

    print()

    print("=" * 60)

    print("Prediction Finished Successfully")

    print("=" * 60)

    print()

    print("Saved Result :")

    print(output_path)

    print()


############################################################
# Entry Point
############################################################

if __name__ == "__main__":

    main()