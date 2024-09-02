import os
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

from config import L, pitch, width


# Transducer locations for rectangular linear array
def x_i(i):
    # Equation for the x-coordinate of the i-th transducer element
    # This is based on the geometry described in your prompt
    return -L / 2 + pitch * (i) + width / 2


def r0(i):
    # Return the center coordinates of the i-th transducer element
    # This replaces the spherical coordinates used in the original paper
    return np.array([x_i(i), 0, 0])


def save_reconstructed_images(reconstructed_images, filename):
    # drive.mount('/content/drive')
    # save_path = '/content/drive/My Drive/' + filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"./outputs/{timestamp}/"
    os.makedirs(folder_name, exist_ok=True)
    save_path = folder_name + filename

    with h5py.File(save_path, "w") as f:
        for key, value in tqdm(reconstructed_images.items(), desc="Saving data"):
            dataset_name = f"sphere_{key[0]}_sir_{key[1]}_bf_{key[2]}_fnumber_{key[3]}_apod_{key[4]}"
            f.create_dataset(dataset_name, data=value, compression="gzip")

    print(f"Data saved to {save_path}")
