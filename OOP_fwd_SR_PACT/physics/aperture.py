import numpy as np


class Aperture:
    def __init__(self, a, b, pitch, N_elements, sampling_rate, center_freq, bandwidth, is_sir, PhysicsParameters):
        self.a = a  # Width of each element
        self.b = b  # Height of each element
        self.L = (N_elements - 1) * pitch + a  # Total length of the transducer
        self.pitch = pitch  # Distance between element center
        self.N_elements = N_elements  # Number of elements
        self.sampling_rate = sampling_rate
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.is_sir = is_sir
        self.PhysicsParameters = PhysicsParameters

    def x_i(self, i):
        """Calculate x-position of i-th element center."""
        return -self.L / 2 + self.pitch * i + self.a / 2

    def r0(self, i):
        """Get position vector of i-th element center."""
        return np.array([self.x_i(i), 0, 0])

    def hEIR(self, frequencies):
        """Calculate electrical impulse response in frequency domain."""
        sigma = self.bandwidth / (2 * np.sqrt(2 * np.log(2)))
        gaussian_response = np.exp(-((frequencies - self.center_freq) ** 2) / (2 * sigma**2))
        return gaussian_response / np.max(gaussian_response)

    def hSIR_f(self, frequencies, position, i):
        """
        Calculate spatial impulse response in frequency domain.
        """
        xn, yn, zn = position

        x_nq = xn - self.x_i(i)
        y_nq = yn

        r_qn = np.linalg.norm(position - self.r0(i))

        sinc_x = np.sinc(self.a * frequencies * x_nq / (self.PhysicsParameters.c0 * r_qn))
        sinc_y = np.sinc(self.b * frequencies * y_nq / (self.PhysicsParameters.c0 * r_qn))

        if self.is_sir:
            h_qs = (
                np.exp(-1j * 2 * np.pi * frequencies * r_qn / self.PhysicsParameters.c0)
                / (2 * np.pi * r_qn)
                * sinc_x
                * sinc_y
            )
        else:
            h_qs = np.exp(-1j * 2 * np.pi * frequencies * r_qn / self.PhysicsParameters.c0) / (2 * np.pi * r_qn)
        return h_qs
