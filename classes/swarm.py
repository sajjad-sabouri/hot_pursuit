import numpy as np
from classes.particle import Particle
from copy import copy
from colorama import Fore
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor


class PSO_Controller:
    def __init__(self, **kwargs):
        self._population_size = kwargs['population_size']
        self._n_dims = kwargs['n_dims']
        self._bounds = kwargs['bounds']
        self._cost_function = kwargs['cost_function']
        self._options = kwargs['options'] if 'options' in kwargs else {'c1': 2, 'c2': 2, 'w': 0.9, 'w_damper': 0.99}
        self._problem_state = kwargs['problem_state'] if 'problem_state' in kwargs else None
        self._best_particle = None
        self.initialize_particles()
        self._cost_history = []
        self.update_global_best()

    def initialize_particles(self):
        self._population = [
            Particle(
                n_dims=self._n_dims,
                cost_function=self._cost_function,
                problem_state=self._problem_state,
                bounds=self._bounds
            )
            for _ in range(self._population_size)
        ]

    def update_global_best(self):
        costs = [particle.get_cost() for particle in self._population]
        self._best_particle = copy(self._population[np.argmin(costs)])
        self._cost_history.append(np.min(costs))

    def step(self):
        self._options['w'] *= self._options['w_damper']

        positions = []
        for particle in self._population:
            positions.append(particle.update_position(self._best_particle, self._options))

        futures = []
        for i in range(len(positions)):
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures.append(executor.submit(self._cost_function, positions[i], self._problem_state))

        costs = []
        for future in futures:
            costs.append(future.result())

        for i in range(len(self._population)):
            self._population[i].apply_new_state(costs[i])

        self.update_global_best()

    def report(self):
        print(f'  Best Cost: {Fore.LIGHTGREEN_EX}{np.round(self._best_particle.get_cost(), 4)}{Fore.RESET}', end='')
        print(
            f'   Best Pos: {{{Fore.GREEN}{",  ".join([str(np.round(self._best_particle.get_position()[0, i], 2)) for i in range(self._best_particle.get_position().shape[1])])}{Fore.RESET}}}')

    def plot_cost_history(self):
        plt.figure(figsize=(19.2, 10.8))
        plt.plot(np.arange(len(self._cost_history)), self._cost_history, '--o', alpha=0.7, color='grey')
        plt.tight_layout()
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.title('Cost History', fontsize=14)

        plt.show()

    def get_best_pos(self):
        return self._best_particle.get_position()