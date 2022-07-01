from classes.particle import Particle
import numpy as np


class Map:
    def __init__(self, **kwargs):
        self._corner_points = kwargs['corners'] if 'corners' in kwargs else None
        self._particles = []

    def initialize_particles(self):
        self._particles.append(
            Particle(
                velocity=np.array([0, 0]),
                coordinates=np.array([0.1, 0.1]),
                power=1,
                step_size=0.05
            )
        )
        self._particles.append(
            Particle(
                velocity=np.array([0, 0]),
                coordinates=np.array([0.9, 0.9]),
                power=0,
                step_size=0.05
            )
        )
