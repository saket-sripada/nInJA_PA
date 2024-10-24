import numpy as np


class PhysicsParameters:
    def __init__(self, c0):
        self.dt = None  # Time step
        self.d_max = None  # Maximum distance for simulation
        self.scan_T = None  # Total scanning time
        self.times = None  # Array of sampling timestamps
        self.size_detThr = None  # Detection threshold size
        self.N_samples = None  # Number of samples
        self.c0 = c0  # Speed of sound in tissue

    def calculate_derived_parameters(self, aperture):
        self.dt = 1.0 / aperture.sampling_rate
        self.d_max = np.sqrt(2) * aperture.L
        self.scan_T = self.d_max / self.c0
        self.N_samples = int(self.scan_T / self.dt)
        self.times = np.linspace(0.0, self.N_samples * self.dt, self.N_samples, endpoint=False)
        self.size_detThr = self.c0 / aperture.sampling_rate


class Sphere:
    def __init__(self, radius, source_fluence, x, y, z, nInJA_muA = 42, muA_bg=0.1, muS_bg=1.0, g_bg=0.9):
        self.radius = radius

        self.position = np.array([x, y, z])
        self.nInJA_muA = nInJA_muA #
        self.muA_bg = muA_bg  # background tissue absorption coefficient
        self.muS_bg = muS_bg # background tissue scattering coefficient
        self.g_fwscat_bg = g_bg  # background tissue g-factor
        self.muEff_bg = np.sqrt(self.muA_bg * (self.muA_bg + self.muS_bg * (1 - self.g_fwscat_bg)))
        self.intensity = source_fluence * self.nInJA_muA * np.exp(-z * self.muEff_bg)


class NInJACluster:
    def __init__(self, center_x, center_z, cluster_size, num_ninjas, detection_threshold):
        self.center = np.array([center_x, 0, center_z])
        self.cluster_size = cluster_size
        self.num_ninjas = num_ninjas
        self.particles = self.generate_particles(detection_threshold)

    def generate_particles(self, detection_threshold):
        particles = []
        for _ in range(self.num_ninjas):
            x = np.random.uniform(self.center[0] - self.cluster_size / 2, self.center[0] + self.cluster_size / 2)
            z = np.random.uniform(self.center[2] - self.cluster_size / 2, self.center[2] + self.cluster_size / 2)
            radius = np.random.uniform(0, detection_threshold / 2)  # Ensure radius is below detection threshold
            intensity = 1.0  # Placeholder intensity; can be adjusted as needed
            particles.append(Sphere(radius, intensity, x, 0, z))
        return particles


class PhantomConfiguration:
    def __init__(self, spheres):
        self.spheres = spheres
        # self.clusters = clusters
