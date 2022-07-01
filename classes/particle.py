from classes.map import Line_With_Vector_And_Point
import numpy as np


class Particle:
    def __init__(self, **kwargs):
        self._velocity = kwargs['velocity'] if 'velocity' in kwargs else None
        self._coordinates = kwargs['coordinates'] if 'coordinates' in kwargs else None
        self._power = kwargs['power'] if 'power' in kwargs else None
        self._step_size = kwargs['step_size'] if 'step_size' in kwargs else None
        self._n_move_vectors = kwargs['n_move_vectors'] if 'n_move_vectors' in kwargs else None
        self._battle = kwargs['battle'] if 'battle' in kwargs else None
        self._move_vectors = []
        self._distances = []
        self.construct_move_vectors()
        self.find_distance_to_boundaries()

    def construct_move_vectors(self):
        base_vector = self._velocity
        theta_array = np.linspace(0, 2 * np.pi, self._n_move_vectors+1)

        self._move_vectors.clear()
        for theta_index in range(theta_array.shape[0] - 1):
            theta = theta_array[theta_index]
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]
            )
            self._move_vectors.append(np.matmul(rotation_matrix, base_vector))

    def find_distance_to_boundaries(self):
        self._distances.clear()

        for move_vector in self._move_vectors:
            distances = np.zeros(len(self._battle.get_map_boundaries()))
            counter = 0
            for boundary in self._battle.get_map_boundaries():
                distance = boundary.find_distance_to_line(
                    Line_With_Vector_And_Point(
                        vector=move_vector,
                        point=self._coordinates
                    )
                )
                distances[counter] = distance if distance is not None else np.inf
                counter += 1

            self._distances.append(np.min(distances))
