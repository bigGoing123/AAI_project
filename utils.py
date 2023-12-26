import os
import random
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image

transform = torchvision.transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_image_and_label(images: np.ndarray):
    for label in range(images.shape[0]):
        return_image = images[label, :, :].copy()
        image = images[label, :, :].copy().reshape((1, 28 * 28))[0].tolist()
        if any(image):
            return return_image, label
    return None


def load_data(data_type: str = "train", is_overwrite=True, is_visualize=False):
    if is_overwrite:
        return_data_list, return_target_list = [], []
    else:
        return np.load("return_data_list.npy"), np.load("return_target_list.npy")

    if data_type == "train":
        path = "mnist/processed_data/train"
        visual_data_path = "visual_data/train"
    elif data_type == "validate":
        path = "mnist/processed_data/val"
        visual_data_path = "visual_data/val"
    else:
        path = "mnist/processed_data/test"
        visual_data_path = "visual_data/test"

    if is_visualize:
        os.makedirs(visual_data_path, exist_ok=True)

    folder_list, image_path_list, image_name_list = os.listdir(path), [], []
    if data_type == "train" or data_type == "validate":
        for folder_number in folder_list:
            for image_path in os.listdir(os.path.join(path, folder_number)):
                image_path_list.append(os.path.join(os.path.join(path, folder_number, image_path)))
                image_name_list.append(image_path)
    else:
        for image_path in folder_list:
            image_path_list.append(os.path.join(os.path.join(path, image_path)))
            image_name_list.append(image_path)

    for image_path, image_name in tqdm(zip(image_path_list, image_name_list),
                                       total=len(image_path_list),
                                       desc="Processing data"):
        return_image, label = get_image_and_label(np.load(image_path))
        return_data_list.append(return_image)
        return_target_list.append(label)
        if is_visualize:
            return_image = return_image > 0
            image = Image.fromarray(return_image)
            if image.mode == "F":
                image = image.convert("RGB")
            image.save(os.path.join(visual_data_path, f"{image_name.replace('.npy', '')}-{data_type}.jpg"))
    np.save("return_data_list.npy", return_data_list)
    np.save("return_target_list.npy", return_target_list)
    return return_data_list, image_name_list


def write_image(image_list, label_list, name_list):
    save_path = "written images"
    os.makedirs(save_path, exist_ok=True)
    for img_name, image, label in tqdm(zip(name_list, image_list, label_list), total=len(label_list), desc="Writing"):
        image = image > 0
        image = Image.fromarray(image)
        if image.mode == "F":
            image = image.convert("RGB")
        image.save(os.path.join(save_path, f"{img_name.replace('.npy', '')}_label-{label}.jpg"))


def set_seed(seed: int = 6666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_image(data_path):
    images_list = []
    for filename in os.listdir(data_path):
        file_contains = os.listdir(os.path.join(data_path, filename))
        for picture_path in tqdm(file_contains, total=len(file_contains), desc="Loading Dataset"):
            images_list.append(transform(os.path.join(data_path, filename, picture_path)))
    return images_list
