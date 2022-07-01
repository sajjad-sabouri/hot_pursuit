from classes.particle import Particle
from classes.boundary import Map_Boundaries
import numpy as np


class Map:
    def __init__(self, **kwargs):
        self._map_boundaries = Map_Boundaries(
            polygons=kwargs['polygons'] if 'polygons' in kwargs else None
        )
        self._particles = []

    def initialize_particles(self):
        self._particles.append(
            Particle(
                velocity=np.array([0, 1]),
                coordinates=np.array([0.1, 0.1]),
                power=1,
                step_size=0.05,
                n_move_vectors=12,
                map=self
            )
        )
        self._particles.append(
            Particle(
                velocity=np.array([-1, 0]),
                coordinates=np.array([0.9, 0.9]),
                power=0,
                step_size=0.05,
                n_move_vectors=12,
                map=self
            )
        )

    def get_boundaries(self):
        return self._map_boundaries.get_boundaries()
