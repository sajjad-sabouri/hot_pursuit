import numpy as np


class Gravity_Driver:
    def __init__(self, **kwargs):
        self._gravity_center = kwargs['gravity_center']
        self._gravity_intensity_center = kwargs['gravity_intensity_center']
        self._gravity_intensity_variance = kwargs['gravity_intensity_variance']

    def bivariate_normal_distribution(self, current_position):
        return 1 / (2 * np.pi * self._gravity_intensity_variance[0] * self._gravity_intensity_variance[1]) * np.exp(
            -0.5 *
            (
                np.power(
                    (current_position[0] - self._gravity_intensity_center[0])/self._gravity_intensity_variance[0],
                    2
                )
                + np.power(
                    (current_position[1] - self._gravity_intensity_center[1])/self._gravity_intensity_variance[1],
                    2
                )
            )
        )

    def calculate_driver_gravity(self, current_position):
        intensity = self.bivariate_normal_distribution(current_position)
        move_vector = self._gravity_center - current_position
        return move_vector / np.linalg.norm(move_vector), intensity

    def get_gravity_center(self):
        return self._gravity_center

    def get_gravity_centers(self):
        return self._gravity_center, self._gravity_intensity_center
