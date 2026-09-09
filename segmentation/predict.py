import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPImageProcessor

from multitask_model import MultiTaskModel


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Multi-Task Face Prediction")
print("=" * 60)
print(f"Device : {device}")


# ============================================================
# Paths
# ============================================================

CHECKPOINT = "checkpoints/best_model.pt"

PROCESSOR_PATH = "../models/clip-vit-base-patch32"

MODEL_PATH = "../models/clip-vit-base-patch32"

RESULT_DIR = Path("results")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Labels
# ============================================================

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
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian"
]


# ============================================================
# Load Processor
# ============================================================

print("\nLoading processor...")

processor = CLIPImageProcessor.from_pretrained(
    PROCESSOR_PATH
)


# ============================================================
# Load Main Model
# ============================================================

print("Loading MultiTask model...")

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
print("MultiTask Model Loaded Successfully")
print("=" * 60)


# ============================================================
# SAM2 - Lazy Loading
# ============================================================

sam2_generator = None


def load_sam2():
    """
    Load SAM2 only when the SAM2 mode is selected.
    """

    global sam2_generator

    if sam2_generator is not None:
        return sam2_generator

    try:

        print()
        print("=" * 60)
        print("Loading SAM2...")
        print("=" * 60)

        from transformers import pipeline

        sam2_generator = pipeline(
            task="mask-generation",
            model="../checkpoints/sam2.1-hiera-large",
            device=0 if torch.cuda.is_available() else -1
        )

        print("SAM2 Loaded Successfully")

        return sam2_generator

    except Exception as e:

        print()
        print("=" * 60)
        print("SAM2 Loading Failed")
        print("=" * 60)
        print(e)

        raise


# ============================================================
# SAM2 Face Extraction
# ============================================================

def extract_sam2_face(image):
    """
    Extract the best face from an image using SAM2.

    Returns:
        PIL.Image or None
    """

    try:

        from face_crop import (
            crop_face_from_mask,
            select_face_mask
        )

        generator = load_sam2()

        image_rgb = image.convert("RGB")

        image_np = __import__("numpy").array(
            image_rgb
        )

        masks = generator(
            image_rgb,
            points_per_batch=64
        )

        if not masks["masks"]:
            return None

        best_mask = select_face_mask(
            image_np,
            masks
        )

        if best_mask is None:
            return None

        if hasattr(best_mask, "cpu"):
            best_mask = best_mask.cpu().numpy()

        face = crop_face_from_mask(
            image_np,
            best_mask
        )

        return face

    except Exception as e:

        print(
            f"SAM2 face extraction error: {e}"
        )

        return None


# ============================================================
# Main Prediction
# ============================================================

@torch.no_grad()
def predict_image(image):
    """
    Predict gender, age and race for a PIL image.
    """

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    # --------------------------------------------------------
    # Synchronize CUDA for accurate timing
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    outputs = model.predict(
        pixel_values
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.time() - start_time
    ) * 1000

    # ========================================================
    # Gender
    # ========================================================

    gender_index = outputs[
        "gender_prediction"
    ].item()

    gender_probability = outputs[
        "gender_probability"
    ].squeeze().item()

    if gender_index == 0:
        gender_confidence = (
            1.0 - gender_probability
        )
    else:
        gender_confidence = (
            gender_probability
        )

    gender_confidence *= 100

    # ========================================================
    # Age
    # ========================================================

    age_index = outputs[
        "age_prediction"
    ].item()

    age_probability = (
        outputs["age_probability"]
        .squeeze()[age_index]
        .item()
        * 100
    )

    # ========================================================
    # Race
    # ========================================================

    race_index = outputs[
        "race_prediction"
    ].item()

    race_probability = (
        outputs["race_probability"]
        .squeeze()[race_index]
        .item()
        * 100
    )

    return {
        "gender": GENDER_LABELS[gender_index],
        "gender_probability": gender_confidence,

        "age": AGE_LABELS[age_index],
        "age_probability": age_probability,

        "race": RACE_LABELS[race_index],
        "race_probability": race_probability,

        "inference_time": inference_time
    }


# ============================================================
# Single Image Prediction
# ============================================================

def single_image_prediction():

    print()
    print("=" * 60)
    print("Single Image Prediction")
    print("=" * 60)

    root = tk.Tk()
    root.withdraw()

    root.attributes(
        "-topmost",
        True
    )

    image_path = filedialog.askopenfilename(
        parent=root,
        title="Select Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    root.destroy()

    if not image_path:

        print("\nNo image selected.")

        return

    if not os.path.exists(image_path):

        print("\nImage not found.")

        return

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as e:

        print(
            f"\nCould not open image: {e}"
        )

        return

    result = predict_image(
        image
    )

    # --------------------------------------------------------
    # Console Result
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print()
    print(f"Image : {image_path}")

    print()

    print(
        f"Gender : {result['gender']}"
    )

    print(
        f"Probability : "
        f"{result['gender_probability']:.2f}%"
    )

    print()

    print(
        f"Age : {result['age']}"
    )

    print(
        f"Probability : "
        f"{result['age_probability']:.2f}%"
    )

    print()

    print(
        f"Race : {result['race']}"
    )

    print(
        f"Probability : "
        f"{result['race_probability']:.2f}%"
    )

    print()

    print(
        f"Inference Time : "
        f"{result['inference_time']:.2f} ms"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Show Result
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image
    )

    plt.axis("off")

    plt.title(
        f"Gender : {result['gender']} "
        f"({result['gender_probability']:.1f}%)\n"
        f"Age : {result['age']} "
        f"({result['age_probability']:.1f}%)\n"
        f"Race : {result['race']} "
        f"({result['race_probability']:.1f}%)",
        fontsize=12
    )

    output_path = (
        RESULT_DIR /
        (
            Path(image_path).stem +
            "_prediction.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("=" * 60)
    print("Prediction Finished Successfully")
    print("=" * 60)

    print()
    print("Saved Result:")
    print(output_path)
    print()


# ============================================================
# Draw Prediction
# ============================================================

def draw_prediction(
    frame,
    result,
    title="Multi-Task Face Prediction"
):
    """
    Draw prediction information on webcam frame.
    """

    overlay = frame.copy()

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    cv2.rectangle(
        overlay,
        (10, 10),
        (450, 195),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65

    thickness = 2

    x = 25

    y = 40

    line_height = 35

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    cv2.putText(
        frame,
        title,
        (x, y),
        font,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    y += line_height

    cv2.putText(
        frame,
        (
            f"Gender: {result['gender']} "
            f"({result['gender_probability']:.1f}%)"
        ),
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    y += line_height

    cv2.putText(
        frame,
        (
            f"Age: {result['age']} "
            f"({result['age_probability']:.1f}%)"
        ),
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Race
    # --------------------------------------------------------

    y += line_height

    cv2.putText(
        frame,
        (
            f"Race: {result['race']} "
            f"({result['race_probability']:.1f}%)"
        ),
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    y += line_height

    cv2.putText(
        frame,
        (
            f"Inference: "
            f"{result['inference_time']:.1f} ms"
        ),
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# Real-Time Camera
# ============================================================

def realtime_prediction(
    use_sam2=False
):

    if use_sam2:

        mode_title = (
            "Real-Time Camera + SAM2"
        )

        print()
        print("=" * 60)
        print("Real-Time Camera + SAM2")
        print("=" * 60)

        try:
            load_sam2()

        except Exception:

            messagebox.showerror(
                "SAM2 Error",
                (
                    "SAM2 could not be loaded.\n\n"
                    "Check the SAM2 model path and "
                    "installation."
                )
            )

            return

    else:

        mode_title = (
            "Real-Time Camera"
        )

        print()
        print("=" * 60)
        print("Real-Time Camera")
        print("=" * 60)

    # ========================================================
    # Camera
    # ========================================================

    cap = cv2.VideoCapture(
        0
    )

    if not cap.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Could not open the camera."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # ========================================================
    # Settings
    # ========================================================

    # Main model prediction interval
    prediction_interval = 3

    # SAM2 is much heavier, so it runs less frequently
    sam2_interval = 30

    frame_counter = 0

    last_result = None

    last_face = None

    prev_time = time.time()

    fps = 0

    # ========================================================
    # Camera Loop
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Could not read camera frame."
                )

                break

            # Mirror
            frame = cv2.flip(
                frame,
                1
            )

            frame_counter += 1

            # ------------------------------------------------
            # Convert current frame to PIL
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            current_image = Image.fromarray(
                rgb_frame
            )

            # =================================================
            # Mode 1:
            # Normal Real-Time Prediction
            # =================================================

            if not use_sam2:

                if (
                    last_result is None
                    or
                    frame_counter % prediction_interval == 0
                ):

                    try:

                        last_result = predict_image(
                            current_image
                        )

                    except Exception as e:

                        print(
                            f"Prediction error: {e}"
                        )

            # =================================================
            # Mode 2:
            # SAM2 + Real-Time Prediction
            # =================================================

            else:

                # ---------------------------------------------
                # Run SAM2
                # ---------------------------------------------

                if (
                    last_face is None
                    or
                    frame_counter % sam2_interval == 0
                ):

                    try:

                        new_face = extract_sam2_face(
                            current_image
                        )

                        if new_face is not None:

                            last_face = new_face

                            # -------------------------------
                            # Prediction on cropped face
                            # -------------------------------

                            last_result = predict_image(
                                last_face
                            )

                    except Exception as e:

                        print(
                            f"SAM2/Prediction error: {e}"
                        )

                # ---------------------------------------------
                # If a face exists, update prediction
                # ---------------------------------------------

                elif (
                    last_face is not None
                    and frame_counter % prediction_interval == 0
                ):

                    try:

                        last_result = predict_image(
                            last_face
                        )

                    except Exception as e:

                        print(
                            f"Prediction error: {e}"
                        )

            # =================================================
            # Draw Result
            # =================================================

            if last_result is not None:

                frame = draw_prediction(
                    frame,
                    last_result,
                    mode_title
                )

            # =================================================
            # Status
            # =================================================

            if use_sam2:

                if last_face is None:

                    cv2.putText(
                        frame,
                        "Searching for face...",
                        (25, 245),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                else:

                    cv2.putText(
                        frame,
                        "Face detected by SAM2",
                        (25, 245),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            # =================================================
            # FPS
            # =================================================

            current_time = time.time()

            elapsed = (
                current_time - prev_time
            )

            if elapsed > 0:
                fps = 1 / elapsed

            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (25, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # =================================================
            # Exit
            # =================================================

            cv2.putText(
                frame,
                "Press Q or ESC to exit",
                (
                    25,
                    frame.shape[0] - 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # =================================================
            # Show
            # =================================================

            cv2.imshow(
                mode_title,
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27
            ):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        for _ in range(3):
            cv2.waitKey(1)

    print()
    print("=" * 60)
    print("Camera Closed")
    print("=" * 60)


# ============================================================
# Main Menu
# ============================================================

def main():

    root = tk.Tk()

    root.title(
        "Multi-Task Face Prediction"
    )

    root.geometry(
        "560x460"
    )

    root.resizable(
        False,
        False
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = tk.Label(
        root,
        text="Multi-Task Face Prediction",
        font=("Arial", 21, "bold")
    )

    title.pack(
        pady=(35, 10)
    )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description = tk.Label(
        root,
        text=(
            "Choose a prediction mode"
        ),
        font=("Arial", 11)
    )

    description.pack(
        pady=(0, 25)
    )

    # --------------------------------------------------------
    # Single Image
    # --------------------------------------------------------

    image_button = tk.Button(
        root,
        text="Single Image Prediction",
        font=("Arial", 12, "bold"),
        width=32,
        height=2,
        command=lambda: run_mode(
            root,
            single_image_prediction
        )
    )

    image_button.pack(
        pady=8
    )

    # --------------------------------------------------------
    # Normal Camera
    # --------------------------------------------------------

    camera_button = tk.Button(
        root,
        text="Real-Time Camera",
        font=("Arial", 12, "bold"),
        width=32,
        height=2,
        command=lambda: run_mode(
            root,
            lambda: realtime_prediction(
                use_sam2=False
            )
        )
    )

    camera_button.pack(
        pady=8
    )

    # --------------------------------------------------------
    # Camera + SAM2
    # --------------------------------------------------------

    sam2_button = tk.Button(
        root,
        text="Real-Time Camera + SAM2",
        font=("Arial", 12, "bold"),
        width=32,
        height=2,
        command=lambda: run_mode(
            root,
            lambda: realtime_prediction(
                use_sam2=True
            )
        )
    )

    sam2_button.pack(
        pady=8
    )

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    exit_button = tk.Button(
        root,
        text="Exit",
        font=("Arial", 11),
        width=15,
        command=root.destroy
    )

    exit_button.pack(
        pady=20
    )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    root.mainloop()


# ============================================================
# Run Mode
# ============================================================

def run_mode(
    root,
    function
):
    """
    Close the menu and run selected mode.
    """

    root.destroy()

    function()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()