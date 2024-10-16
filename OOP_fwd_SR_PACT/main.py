import numpy as np
import torch
from imaging.reconstruction import ImageReconstructor, ReconstructionParameters
from imaging.visualizer import Visualizer
from physics.aperture import Aperture
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
    physics_params = PhysicsParameters(c0=1540)

    # Initialize aperture for L38xP 5-10 MHz Tx
    aperture = Aperture(
        a=0.2e-3,  # width of 0.2 mm
        b=5e-3,  # height of 5 mm
        # total length of 38.2 mm
        pitch=0.3e-3,  # pitch of 0.3 mm
        N_elements=128,  # 128 elements for recieving
        sampling_rate=48e6,  # sampling rate of 48 MHz although system can capture up to 192 MHz
        center_freq=7.5e6,  # center frequency of 7.5 MHz
        bandwidth=3e6,  # bandwidth of 3 MHz
        is_sir=True,  # finite spatial impulse response T/F toggle moved from recon
        PhysicsParameters=physics_params,
    )

    # Calculate derived physics parameters
    physics_params.calculate_derived_parameters(aperture)

    # Initialize reconstruction parameters
    recon_params = ReconstructionParameters(
        is_sir_list=[True, False],
        recon_methods=["DMAS", "DAS"],
        f_numbers=np.arange(0, 1.7, 0.4),
        apodisation_types=["None"],  # , "hann", "hamming"],
    )

    pressure_data = {}

    # Define sphere and cluster positions

    # want RHS of phantom to have 3 larger spheres from center to lateral extreme
    # and LHS to have clusters of ID size but having tiny spheres
    x_positions = np.linspace(0, aperture.L / 2, 3)
    # z_positions = np.arange(10, 31, 10) * 1e-3 # 3 depths
    z_positions = np.linspace(0.25 * aperture.L, 0.75 * aperture.L, 3)

    for r in [0.5, 0.5e-1]:  # , 0.5e-2, 0.5e-3]:
        # Set properties of spheres
        # intensity is missing a way for G fCTOR, THE MU_A OF PARTICLE AND THE MU_EFF OF THE MEDIUM
        spheres = [
            Sphere(radius=r * 1e-3, intensity=np.exp(-z), x=x, y=0, z=z) for x in x_positions for z in z_positions
        ]

        # Initialize phantom configuration
        phantom_config = PhantomConfiguration(spheres)

        # Create pressure data generator
        generator = PressureDataGenerator(phantom_config, aperture, physics_params)

        # Generate pressure data
        pressure_data[r] = generator.generate_pressure_data()

        # Create image reconstructor
        field_of_view = np.array([-aperture.L / 2, aperture.L / 2, 0, 0, 0, aperture.L])
        reconstructor = ImageReconstructor(aperture, physics_params, field_of_view)

        # Reconstruct images
        reconstructed_images = {}
        for is_sir in recon_params.is_sir_list:
            for recon_method in recon_params.recon_methods:
                for f_number in recon_params.f_numbers:
                    for apodisation in recon_params.apodisation_types:
                        recon_config = {
                            "is_sir": is_sir,
                            "radius": r,
                            "method": recon_method,
                            "f_number": f_number,
                            "apodisation": apodisation,
                        }
                        reconstructed_images[tuple(recon_config.values())] = reconstructor.reconstruct_image(
                            time_series_data=torch.tensor(pressure_data[r], dtype=torch.float32),
                            bf=recon_method,
                            spacing_in_m=aperture.L / aperture.N_elements,
                            fnumber=f_number,
                            apodisation=apodisation,
                            torch_device=torch.device("cpu")
                        )

    # Create visualizer
    visualizer = Visualizer(phantom_config, aperture)

    # Plot and save results
    visualizer.plot_comparison(pressure_data, reconstructed_images)
    save_data_h5("./outputs/", pressure_data, reconstructed_images)


if __name__ == "__main__":
    main()
