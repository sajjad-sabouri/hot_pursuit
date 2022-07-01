import numpy as np


class Particle:
    def __init__(self, **kwargs):
        self._velocity = kwargs['velocity'] if 'velocity' in kwargs else None
        self._coordinates = kwargs['coordinates'] if 'coordinates' in kwargs else None
        self._power = kwargs['power'] if 'power' in kwargs else None
        self._step_size = kwargs['step_size'] if 'step_size' in kwargs else None
        self._n_move_vectors = kwargs['n_move_vectors'] if 'n_move_vectors' in kwargs else None
        self._move_vectors = []
        self.construct_move_vectors()

    def construct_move_vectors(self):
        base_vector = self._velocity
        tetas = np.linspace(0, 2 * np.pi, self._n_move_vectors)

        self._move_vectors.clear()
        for teta in tetas:
            rotation_matrix = np.array(
                [np.cos(teta), -np.sin(teta)],
                [np.sin(teta), np.cos(teta)]
            )
            self._move_vectors.append(np.matmul(rotation_matrix, base_vector))
