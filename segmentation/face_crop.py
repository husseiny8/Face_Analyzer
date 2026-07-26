from PIL import Image
import numpy as np
from transformers import CLIPProcessor
from transformers import CLIPModel
import torch


def geometry_filter(
    image,
    masks,
    min_area_ratio=0.01,
    max_area_ratio=0.65,
    min_size=40,
    min_aspect=0.55,
    max_aspect=1.80,
    max_center_ratio=0.35,
):

    H, W = image.shape[:2]
    image_area = H * W
    image_center = np.array([W / 2, H / 2])
    max_center_distance = np.sqrt(W**2 + H**2) * max_center_ratio
    filtered_masks = []

    for mask in masks["masks"]:

        m = mask.cpu().numpy().astype(bool)
        area = m.sum()

        # Area
        if area < image_area * min_area_ratio:
            continue

        if area > image_area * max_area_ratio:
            continue

        # Bounding Box
        ys, xs = np.where(m)

        if len(xs) == 0:
            continue

        x1 = xs.min()
        x2 = xs.max()

        y1 = ys.min()
        y2 = ys.max()

        width = x2 - x1 + 1
        height = y2 - y1 + 1

        # Size
        if width < min_size:
            continue

        if height < min_size:
            continue

        # Aspect Ratio
        ratio = width / height

        if ratio < min_aspect:
            continue

        if ratio > max_aspect:
            continue

        # Border Filter
        border_hits = 0

        if y1 == 0:
            border_hits += 1

        if y2 >= H - 1:
            border_hits += 1

        if x1 == 0:
            border_hits += 1

        if x2 >= W - 1:
            border_hits += 1

        if border_hits >= 2:
            continue

        # Center Distance
        center = np.array([(x1 + x2) / 2,(y1 + y2) / 2])

        distance = np.linalg.norm(center - image_center)

        if distance > max_center_distance:
            continue

        filtered_masks.append(mask)

    return filtered_masks


def crop_face_from_mask(
    image,
    mask,
    padding=0.15,
    background_color=(0, 0, 0)
):

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

    # candidate_masks = geometry_filter(
    #     image,
    #     masks
    # )
    #
    # if len(candidate_masks) == 0:
    #     return None

    best_score = -float("inf")
    best_mask = None

    # for mask in candidate_masks:
    for mask in masks['masks']:
        cropped = crop_face_from_mask(image,mask.cpu().numpy())
        # plt.imshow(cropped)
        # plt.show()
        score = similarity(cropped)

        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        return None

    return best_mask