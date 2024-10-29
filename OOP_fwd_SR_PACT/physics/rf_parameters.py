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
    def __init__(self, radius, source_fluence, x, y, z, nInJA_muA, muA_bg=0.1, muS_bg=1.0, g_bg=0.9):
        
        self.radius = radius
        self.position = np.array([x, y, z])
        self.nInJA_muA = nInJA_muA  #
        self.muA_bg = muA_bg  # background tissue absorption coefficient
        self.muS_bg = muS_bg  # background tissue scattering coefficient
        self.g_fwscat_bg = g_bg  # background tissue g-factor
        self.muEff_bg = np.sqrt(self.muA_bg * (self.muA_bg + self.muS_bg * (1 - self.g_fwscat_bg)))
        self.volume = (4 / 3) * np.pi * (self.radius**3)
        # Scale intensity by volume
        # Calculate source_fluence to achieve desired final intensity
        # Work backwards to get required source_fluence
        # Since: final_intensity = source_fluence * nInJA_muA * volume * exp(-z*muEff)
        # We solve for source_fluence
        self._intensity = source_fluence * self.nInJA_muA * np.exp(-z * self.muEff_bg)

    @property
    def intensity(self):
        return self._intensity


class NInJACluster:
    def __init__(self, center_x, center_z, cluster_radius, num_ninjas, source_fluence, nInJA_muA, extrusion_threshold):
        self.center = np.array([center_x, 0, center_z])
        self.cluster_radius = cluster_radius
        self.num_ninjas = int(num_ninjas)
        self.source_fluence = source_fluence
        self.nInJA_muA = nInJA_muA
        self.total_volume = (4 / 3) * np.pi * (self.cluster_radius**3)
        # Scale individual sphere intensities by their fraction of total volume
        self.volume_per_sphere = self.total_volume / self.num_ninjas

        self.particles = self.generate_particles(extrusion_threshold)

    def generate_particles(self, extrusion_threshold):
        particles = []
        for _ in range(self.num_ninjas):
            # Generate random position within sphere using spherical coordinates
            theta = 2 * np.pi * np.random.random()
            # phi = np.arccos(2 * np.random.random() - 1) # phi for 3D
            r = self.cluster_radius * np.sqrt(np.random.random())  # np.cbrt() for 3D

            # Convert to Cartesian coordinates
            x = self.center[0] + r * np.cos(theta)  # * np.sin(phi) # for phi 3D
            z = self.center[2] + r * np.sin(theta)  # * np.sin(phi) # for phi 3D

            # Create sphere with parameters inherited from cluster
            particle = Sphere(
                radius=extrusion_threshold / 2,
                source_fluence=self.source_fluence * (self.volume_per_sphere/self.total_volume),  # scale by volume fraction,
                x=x,
                y=0,
                z=z,
                nInJA_muA=self.nInJA_muA,
            )
            particles.append(particle)
        return particles

    @property
    def total_intensity(self):
        return sum(sphere.intensity for sphere in self.particles)


class PhantomConfiguration:
    def __init__(self, spheres, clusters):
        self.spheres = spheres
        self.clusters = clusters

    def all_sources(self):
        """Returns a unified list of all sources (spheres) regardless of origin"""
        all_spheres = []
        # Add individual spheres
        all_spheres.extend(self.spheres)
        # Add spheres from clusters
        for cluster in self.clusters:
            all_spheres.extend(cluster.particles)
        return all_spheres
