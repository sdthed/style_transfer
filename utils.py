import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def make_dirs(path):
    """ Creates direcories if these don't already exist """
    directory, _ = os.path.split(path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)


def resize(im: Image.Image, max_dim: int) -> Image.Image:
    """ resize image by max dimension """
    im_shape = np.array(im).shape[:-1][::-1]  # (width, height)
    scale = max_dim / np.max(im_shape)
    new_shape = tuple([int(x * scale) for x in im_shape])
    return im.resize(new_shape)


def load_im(
    im_path: str, max_dim: Optional[int] = None, normalize: bool = True
) -> np.ndarray:
    """ read, reshape, and normalize image """
    im = Image.open(im_path)
    if max_dim is not None:
        im = resize(im, 200)
    return np.array(im) / 255.0 if normalize else np.array(im)


def save_im(image: np.ndarray, path: str):
    make_dirs(path)
    im2write = (np.squeeze(image) * 255).astype(np.uint8)
    plt.imsave(path, im2write)


def preview(*images):
    _, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    ax = ax if len(images) > 1 else [ax]
    for i, im in enumerate(images):
        ax[i].imshow(im)

    plt.show()
