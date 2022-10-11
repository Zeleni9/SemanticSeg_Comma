import cv2
import numpy as np
from LitModel import *
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def pad_to_multiple(x, k=32):
    return int(k * (np.ceil(x / k)))


def decode_segmap(image, nc=6):
    label_colors = np.array(
        [
            (64, 32, 32),  # 0 = road (all parts, anywhere nobody would look at you funny for driving)
            (255, 0, 0),  # 1 = lane markings (don't include non lane markings like turn arrows and crosswalks)
            (204, 0, 255),  # 2 =  my car (and anything inside it, including wires, mounts, etc. No reflections)
            (128, 128, 96),  # 3 =  undrivable
            (0, 255, 102),  # 4 = movable (vehicles and people/animals)
            (0, 0, 0),
        ]
    )  # 5 = padded part
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask, width, height):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    image = cv2.resize(image, (width, height))
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def get_preprocessing(preprocessing_fn: Callable):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def get_valid_transforms(height: int = 437, width: int = 582):
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=1.0),
            A.PadIfNeeded(
                pad_to_multiple(height), pad_to_multiple(width), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
            ),
        ],
        p=1.0,
    )
