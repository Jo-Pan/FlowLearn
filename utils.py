import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import base64
import numpy as np


def load_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def display_image(file_path):
    img = mpimg.imread(file_path)
    imgplot = plt.imshow(img)
    plt.show()


def load_image(img_ids, root_path):
    if isinstance(img_ids, str):
        img_ids = [img_ids]
    images = []
    image_paths = []
    for img_id in img_ids:
        image_path = os.path.join(root_path, img_id)
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        image_paths.append(image_path)

    return images, image_paths


def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image, mime_type


def set_random_seed(seed_number):
    # position of setting seeds also matters
    os.environ["PYTHONHASHSEED"] = str(seed_number)
    np.random.seed(seed_number)
