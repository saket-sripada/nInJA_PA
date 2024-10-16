import numpy as np
import torch


class ReconstructionParameters:
    def __init__(self, is_sir_list, recon_methods, f_numbers, apodisation_types):
        self.is_sir_list = is_sir_list
        self.recon_methods = recon_methods
        self.f_numbers = f_numbers
        self.apodisation_types = apodisation_types


class ImageReconstructor:
    def __init__(self, aperture, physics_params):
        self.aperture = aperture
        self.physics_params = physics_params

    def get_apodisation_factor(
        apodization_method: str = "box",
        dimensions: tuple = None,
        n_sensor_elements=None,
        device: torch.device = "cpu",
    ) -> torch.tensor:
        """
        Construct apodization factors according to `apodization_method` [hann, hamming or box apodization (default)]
        for given dimensions and `n_sensor_elements`.

        :param apodization_method: (str) Apodization method, one of hann, hamming and box (default)
        :param dimensions: (tuple) size of each dimension of reconstructed image as int, might have 2 or 3 entries.
        :param n_sensor_elements: (int) number of sensor elements
        :param device: (torch device) PyTorch tensor device
        :return: (torch tensor) tensor with apodization factors which can be multipied with DAS values
        """

        if dimensions is None or n_sensor_elements is None:
            raise AttributeError("dimensions and n_sensor_elements must be specified and not be None")

        # hann window
        if apodization_method == "hann":
            hann = torch.hann_window(n_sensor_elements, device=device)
            output = hann.expand(dimensions + (n_sensor_elements,))
        # hamming window
        elif apodization_method == "hamming":
            hamming = torch.hamming_window(n_sensor_elements, device=device)
            output = hamming.expand(dimensions + (n_sensor_elements,))
        # box window apodization as default
        else:
            output = torch.ones(dimensions + (n_sensor_elements,), device=device)

        return output

    def reconstruct_image(
        self,
        time_series_data: torch.tensor,
        bf: str,
        sensor_positions: torch.tensor,
        field_of_view_voxels: np.ndarray,
        spacing_in_m: float,
        speed_of_sound_in_m_per_s: float,
        time_spacing_in_s: float,
        torch_device: torch.device,
        fnumber: float,
        apodisation,
    ) -> torch.tensor:

        n_sensor_elements = time_series_data.shape[0]
        x_dim = field_of_view_voxels[1] - field_of_view_voxels[0]
        y_dim = field_of_view_voxels[3] - field_of_view_voxels[2]
        z_dim = field_of_view_voxels[5] - field_of_view_voxels[4]
        output = torch.zeros((x_dim, 1, z_dim), dtype=torch.float32, device=torch_device)

        xrmg, yrmg, zrmg, jj = torch.meshgrid(
            (
                torch.arange(field_of_view_voxels[0], field_of_view_voxels[1], device=torch_device)
                if (field_of_view_voxels[1] - field_of_view_voxels[0]) >= 1
                else torch.arange(1, device=torch_device)
            ),
            (
                torch.arange(field_of_view_voxels[2], field_of_view_voxels[3], device=torch_device)
                if (field_of_view_voxels[3] - field_of_view_voxels[2]) >= 1
                else torch.arange(1, device=torch_device)
            ),
            (
                torch.arange(field_of_view_voxels[4], field_of_view_voxels[5], device=torch_device)
                if (field_of_view_voxels[5] - field_of_view_voxels[4]) >= 1
                else torch.arange(1, device=torch_device)
            ),
            torch.arange(n_sensor_elements, device=torch_device),
            indexing="ij",
        )

        jj = jj.long()
        delays = torch.sqrt(
            (yrmg * spacing_in_m - sensor_positions[:, 1][jj]) ** 2
            + (xrmg * spacing_in_m - sensor_positions[:, 0][jj]) ** 2
            + (zrmg * spacing_in_m - sensor_positions[:, 2][jj]) ** 2
        ) / (speed_of_sound_in_m_per_s * time_spacing_in_s)

        # perform index validation
        invalid_indices = torch.where(torch.logical_or(delays < 0, delays >= float(time_series_data.shape[1])))
        torch.clip_(delays, min=0, max=time_series_data.shape[1] - 1)

        # interpolation of delays
        lower_delays = (torch.floor(delays)).long()
        upper_delays = lower_delays + 1
        torch.clip_(upper_delays, min=0, max=time_series_data.shape[1] - 1)
        lower_values = time_series_data[jj, lower_delays]
        upper_values = time_series_data[jj, upper_delays]
        values = lower_values * (upper_delays - delays) + upper_values * (delays - lower_delays)

        values[invalid_indices] = 0  # created issues in the past

        if apodisation is not None:
            xdim, ydim, zdim = (
                field_of_view_voxels[1] - field_of_view_voxels[0],
                field_of_view_voxels[3] - field_of_view_voxels[2],
                field_of_view_voxels[5] - field_of_view_voxels[4],
            )
            if xdim == 0:
                xdim = 1
            if ydim == 0:
                ydim = 1
            if zdim == 0:
                zdim = 1

            apo_funct = ImageReconstructor.get_apodisation_factor(
                apodization_method=apodisation,
                dimensions=(xdim, ydim, zdim),
                n_sensor_elements=n_sensor_elements,
                device=torch_device,
            )
            values = values * apo_funct

        # Add fNumber
        if fnumber > 0:
            values[
                torch.where(
                    torch.logical_not(
                        torch.abs(xrmg * spacing_in_m - sensor_positions[:, 0][jj])
                        < (zrmg * spacing_in_m - sensor_positions[:, 1][jj]) / fnumber / 2
                    )
                )
            ] = 0

        # DMAS section below
        if bf == "DAS":
            return values.sum(dim=-1).squeeze(1).cpu().numpy().T

        elif bf == "DMAS":
            # print('multiplying delays for the DMAS method')
            for x in range(x_dim):
                # Remove yy from meshgrid since y_dim is 1
                zz, nn, mm = torch.meshgrid(
                    torch.arange(z_dim, device=torch_device),
                    torch.arange(n_sensor_elements, device=torch_device),
                    torch.arange(n_sensor_elements, device=torch_device),
                    indexing="ij",
                )
                # Adjust indexing to account for y_dim being 1
                multiplied_signals = values[x, 0, zz, nn] * values[x, 0, zz, mm]
                multiplied_signals = torch.sign(multiplied_signals) * torch.sqrt(torch.abs(multiplied_signals))
                output[x, 0] = torch.triu(multiplied_signals, diagonal=1).sum(dim=(-1, -2))

            """
        if signed_dmas:
            output *= torch.sign(torch.sum(values, dim=3))
            """

            return output.squeeze().cpu().numpy().T
        else:
            print("unknown recon method")
            return None
