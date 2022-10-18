import numpy as np
from swarm import PSO_Controller


def cost_function_sample_01(position):
    return np.sum(np.power(position, 3))


def cost_function_rastrigin(position, A=10):
    return A * position.shape[1] + np.sum(np.power(position, 2) - A * np.cos(2 * np.pi * position))


def cost_function_ackley(position):
    return \
        -20 * np.exp(
            -0.2 * np.sqrt(0.5 * (np.sum(np.power(position, 2))))
        ) - np.exp(
            0.5 * (np.cos(2 * np.pi * position[0, 0]) + np.cos(2 * np.pi * position[0, 1]))
        ) + np.exp(1) + 20


def cost_function_beale(position):
    return np.power(1.5 - position[0, 0] + position[0, 0] * position[0, 1], 2) + \
           np.power(2.25 - position[0, 0] + position[0, 0] * np.power(position[0, 1], 2), 2) + \
           np.power(2.625 - position[0, 0] + position[0, 0] * np.power(position[0, 1], 3), 2)


if __name__ == '__main__':
    population_size = 200
    n_dims = 2
    lower_bounds = np.ones(n_dims) * -4.5
    upper_bounds = np.ones(n_dims) * 4.5
    bounds = {
        'lower_bounds': lower_bounds.reshape(1, -1),
        'upper_bounds': upper_bounds.reshape(1, -1)
    }

    options = {
        'c1': 2,
        'c2': 2,
        'w': 0.9,
        'w_damper': 0.99
    }

    controller = PSO_Controller(
        population_size=population_size,
        n_dims=n_dims,
        bounds=bounds,
        cost_function=cost_function_ackley,
        options=options
    )

    n_iterations = 200
    n_digits = len(str(n_iterations))
    for i in range(n_iterations):
        print(f'Iteration {str(i).zfill(len(str(i)) + n_digits - len(str(i)))} => ', end='')
        controller.step()
        controller.report()
    controller.plot_cost_history()
