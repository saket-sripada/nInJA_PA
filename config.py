import numpy as np
import torch

dist_unit_conv = 1  # e-3 # sticking with m
time_unit_conv = 1  # e-6 # sticking with s
# ----------------------------------------------
# transducer properties
# ----------------------------------------------
a = 0.2 * dist_unit_conv  # trying change to m from mm (width of each element)
b = (
    5 * dist_unit_conv
)  # trying change to m from mm (this is the dimension orthogonal to the imaging plane and the wider
# this is the better the focussing of the beam in the imagin gplane)
L = 40 * dist_unit_conv  # trying change to m from 38.2 mm (length of the transducer)
pitch = 0.3 * dist_unit_conv  # trying change to m from mm
width = 0.2 * dist_unit_conv  # trying change to m from mm
N_elements = 128  # Number of recieve transducer elements, 256 total
center_freq = 8e6  # operating center freq of the 5-10 MHz Tx
bandwidth = 3e6  # +/- 3dB cutoff
sampling_rate = 96 / time_unit_conv  # trying change to [Hz] from MHz...
# for the 8MHz Tx based on spec sheet but can dial to 96MHz or 192MHz Tx

# ----------------------------------------------
# Define the numerical phantom
# ----------------------------------------------
# This corresponds to the phantom description in Section 3.1.1
num_spheres = 3
sphere_value = 42  # [a.u.]
dt = 1.0 / sampling_rate
c0 = 1.54 / dist_unit_conv  # trying change to m/s from mm/us
d_max = np.sqrt(2) * L  # so ensure not setting sphere beyond 4cm depth
scan_T = d_max / c0
N_samples = int(scan_T / dt)  # Full-time sampling -- covers depth
times = np.linspace(0.0, N_samples * dt, N_samples, endpoint=False)
delL = L / num_spheres

# --------------------------------------------------
# resolution phantom params
# --------------------------------------------------
x_positions = np.arange(-L / 2 + delL / 2, L / 2, delL)
y_positions = np.zeros(1)
z_positions = np.arange(10, 31, 10) * dist_unit_conv  # 10mm to 31mm at 10mm steps

sph_radii = [1.542, 0.42, 0.21] * dist_unit_conv
Tsir = [True, False]  # modeling TX as point-like vs finite numerical aperture
recon_meth = ["DMAS", "DAS"]
f_values = np.array(range(0, 10, 2)) / 10  # f#s 0 : 0.2 : 1
apodisation_types = ["None", "hann", "hamming"]

# speed_of_sound_in_m_per_s = c0# # m/s from mm/us
# time_spacing_in_s = dt#*1e-6 # s from us
torch_device = torch.device("cpu")
