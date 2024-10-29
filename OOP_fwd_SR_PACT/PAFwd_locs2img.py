import numpy as np
import torch
from imaging.reconstruction import ImageReconstructor, ReconstructionParameters
from imaging.visualizer import Visualizer
from physics.aperture import Aperture
from physics.noise_models import AcousticNoise
from physics.pressure_generator import PressureDataGenerator
from physics.rf_parameters import (
    NInJACluster,
    PhantomConfiguration,
    PhysicsParameters,
    Sphere,
)
from utils.io_ops import save_data_h5


def main():
    # Initialize physics parameters
    physics_params = PhysicsParameters(c0=1.540 * 1e3)

    # Initialize aperture for L38xP 5-10 MHz Tx
    aperture = Aperture(
        a=0.2 * 1e-3,  # width of 0.2 mm
        b=5 * 1e-3,  # height of 5 mm
        # total length of 38.2 mm
        pitch=0.3 * 1e-3,  # pitch of 0.3 mm
        N_elements=128,  # 128 elements for recieving
        sampling_rate=48 * 1e6,  # sampling rate of 48 MHz although system can capture up to 192 MHz
        center_freq=7.5 * 1e6,  # center frequency of 7.5 MHz
        bandwidth=3 * 1e6,  # bandwidth of 3 MHz
        is_sir=True,  # finite spatial impulse response T/F toggle moved from recon
        PhysicsParameters=physics_params,
    )

    # Calculate derived physics parameters
    physics_params.calculate_derived_parameters(aperture)

    # Define sphere and cluster positions

    # want RHS of phantom to have 3 larger spheres from center to lateral extreme
    # and LHS to have clusters of ID size but having tiny spheres

    # x_positions = np.arange(2, 18, 5) * 1e-3 # 3 positions
    x_positions = np.linspace(0.1 * (aperture.L / 2), 0.9 * (aperture.L / 2), 3)
    # z_positions = np.arange(10, 31, 10) * 1e-3 # 3 depths
    z_positions = np.linspace(0.25 * aperture.L, 0.75 * aperture.L, 3)

    # Set properties of spheres for RHS of phantom

    source_fluence = 4.2

    spheres = [
        Sphere(
            radius=0.3e-3,  # 0.3mm radius to match lambda/2 resolution
            source_fluence=source_fluence,  # placeholder intensity replace with SIMPA fluence
            x=x,
            y=0,  # y=0 for now, as we are working in 2D
            z=z,
            nInJA_muA=4.2,  # set particle abs_coeff = ? number of initial photons per Joule absorbed ?
            muA_bg=1,  # set background tissue absorption coefficient in mm^-1
            muS_bg=10,  # set background scattering coefficient in mm^-1
            g_bg=0.9,  # set g-factor of tissue
        )
        for x in x_positions
        for z in z_positions
    ]

    conc = 1  # number density of nanoscopic IcG JAggregates in a microscopic cluster
    # Set properties of clusters for LHS
    clusters = [
        NInJACluster(
            center_x=-sphere.position[0],  # mirror position
            center_z=sphere.position[2],  # same z position
            cluster_radius=sphere.radius,  # same radius as corresponding sphere
            num_ninjas=int(conc * 100),  # number of nanospheres in cluster, 100 for testing needs to be 1e6
            source_fluence=source_fluence,  # same as spheres
            nInJA_muA=sphere.nInJA_muA,  # same as spheres
            extrusion_threshold=1e-6,  # 1 micron for now so test but should be 1nm
        )
        for sphere in spheres
    ]

    # Initialize phantom configuration
    phantom_config = PhantomConfiguration(spheres, clusters)

    # Create noise model with custom parameters
    acoustic_noise = AcousticNoise(snr_db=25, speckle_std=0.05, noise_type="nakagami")

    # Create pressure data generator
    generator = PressureDataGenerator(phantom_config, aperture, physics_params, acoustic_noise)

    # Generate pressure data
    pressure_data = generator.generate_pressure_data()

    # Create image reconstructor
    # Initialize reconstruction parameters
    recon_params = ReconstructionParameters(recon_methods="DMAS", f_numbers=0.42, apodisation_types="None")

    field_of_view = np.array([-aperture.L / 2, aperture.L / 2, 0, 0, 0, aperture.L])

    reconstructor = ImageReconstructor(aperture, physics_params, field_of_view, acoustic_noise)

    # Reconstruct image

    reconstructed_image = reconstructor.reconstruct_image(
        time_series_data=torch.tensor(pressure_data, dtype=torch.float32),
        bf=recon_params.recon_methods,
        spacing_in_m=aperture.L / aperture.N_elements,
        fnumber=recon_params.f_numbers,
        apodisation=recon_params.apodisation_types,
        torch_device=torch.device("cpu"),
    )

    # Create visualizer
    visualizer = Visualizer(phantom_config, aperture, recon_params, acoustic_noise)

    # simple example visualization of imageing model
    visualizer.plot_GT_RF_BF(pressure_data, reconstructed_image)
    # Plot and save results
    # save_data_h5("./outputs/", pressure_data, recon_params)


if __name__ == "__main__":
    main()
