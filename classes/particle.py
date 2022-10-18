import numpy as np
from copy import copy


class Particle:
    def __init__(self, **kwargs):
        self._n_dims = kwargs['n_dims']
        self._bounds = kwargs['bounds']
        self._cost_function = kwargs['cost_function']
        self._problem_state = kwargs['problem_state'] if 'problem_state' in kwargs else None
        self._position = self.initialize_random_position()
        self._cost = self._cost_function(self._position, self._problem_state)
        self._velocity = np.zeros((1, self._n_dims))
        self._best_position = copy(self._position)
        self._best_cost = self._cost

    def initialize_random_position(self):
        return self._bounds['lower_bounds'] + np.random.rand(1, self._n_dims) * (self._bounds['upper_bounds'] - self._bounds['lower_bounds'])

    def update_best_position(self):
        if self._cost < self._best_cost:
            self._best_cost = copy(self._cost)
            self._best_position = copy(self._position)

    def update_position(self, global_best_particle, options):

        # deduct new position
        new_velocity = \
            options['c1'] * (global_best_particle.get_position() - self._position) + \
            options['c2'] * (self._best_position - self._position) + \
            options['w'] * self._velocity

        new_position = self._position + new_velocity

        # modifying position and velocity according to bounds
        new_velocity[new_position > self._bounds['upper_bounds']] = 0
        new_velocity[new_position < self._bounds['lower_bounds']] = 0
        new_position[new_position > self._bounds['upper_bounds']] = self._bounds['upper_bounds'][new_position > self._bounds['upper_bounds']]
        new_position[new_position < self._bounds['lower_bounds']] = self._bounds['lower_bounds'][new_position < self._bounds['lower_bounds']]

        self._temp_position = new_position
        self._temp_velocity = new_velocity

        return new_position

    def apply_new_state(self, new_cost):

        if new_cost < self._cost:
            self._cost = new_cost
            self._position = copy(self._temp_position)
            self._velocity = copy(self._temp_velocity)
            self.update_best_position()

    def get_cost(self):
        return self._cost

    def get_position(self):
        return self._position
