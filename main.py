import itertools

import numpy as np
import torch
from tqdm import tqdm

from config import (
    L,
    N_elements,
    Tsir,
    a,
    apodisation_types,
    b,
    c0,
    dt,
    f_values,
    recon_meth,
    sph_radii,
    torch_device,
)
from fwd_model.generate_pressure_data import generate_pressure_data
from fwd_model.inverse import reconstruct_image
from fwd_model.io import save_data_h5
from fwd_model.utils import x_i


def main():
    # generate analytical pressure data
    superposition_data = {}
    P0gen_combos = list(itertools.product(sph_radii, Tsir))
    for sphere_radius, is_sir in tqdm(P0gen_combos):
        superposition_data[(sphere_radius, is_sir)] = generate_pressure_data(sphere_radius, is_sir)

    # reconstruct images from generated pressure data
    reconstructed_images = {}
    P0recon_combos = list(itertools.product(sph_radii, Tsir, f_values, apodisation_types, recon_meth))
    spacing_in_m = L / N_elements
    field_of_view = np.array([-L / 2, L / 2, 0, 0, 0, L])  # *dist_unit_conv # mm from m
    field_of_view_voxels = np.round(field_of_view / spacing_in_m).astype(int)
    sensor_positions = torch.tensor(np.array([[x_i(i), 0, 0] for i in range(N_elements)]), dtype=torch.float32)

    for sphere_radius, is_sir, f, apodisation, bf in tqdm(P0recon_combos):
        time_series_data = torch.tensor(superposition_data[(sphere_radius, is_sir)], dtype=torch.float32)
        reconstructed_images[(sphere_radius, is_sir, f, apodisation, bf)] = reconstruct_image(
            time_series_data=time_series_data,
            sphere_radius=sphere_radius,
            is_sir=is_sir,
            bf=bf,
            sensor_positions=sensor_positions,
            field_of_view_voxels=field_of_view_voxels,
            spacing_in_m=spacing_in_m,
            speed_of_sound_in_m_per_s=c0,
            time_spacing_in_s=dt,
            torch_device=torch_device,
            fnumber=f,
            apodisation=apodisation,
        )

    # export / store pressure data
    # save_reconstructed_images(reconstructed_images)
    # save_reconstructed_images(superposition_data)
    save_data_h5(reconstructed_images, superposition_data, a, b)

    # load analytical pressure data, reconstructions

    # visually compare reconstructions

    # save_reconstructed_images(reconstructed_images, "reconstructed_images.h5")


if __name__ == "__main__":
    main()
