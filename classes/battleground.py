from classes.particle import Particle
from classes.map import Map_Boundaries
import numpy as np


class Battleground:
    def __init__(self, **kwargs):
        self._map_boundaries = Map_Boundaries(
            polygons=kwargs['polygons'] if 'polygons' in kwargs else None
        )
        self._particles = []

    def initialize_sample_particles(self):
        self._particles.append(
            Particle(
                velocity=np.array([0, 1]),
                coordinates=np.array([0.1, 0.1]),
                power=1,
                step_size=0.05,
                n_move_vectors=12,
                battle=self
            )
        )

        self._particles.append(
            Particle(
                velocity=np.array([-1, 0]),
                coordinates=np.array([0.9, 0.9]),
                power=0,
                step_size=0.05,
                n_move_vectors=12,
                battle=self
            )
        )

    def get_map_boundaries(self):
        return self._map_boundaries.get_boundaries()
