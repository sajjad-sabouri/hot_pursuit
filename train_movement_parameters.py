from classes.vector_player import Vector_Player
from classes.battleground import Battleground
from classes.weight_functions import WeightFunctions
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
import numpy as np
import json
from matplotlib import pyplot as plt
import pickle


with open('configs/polygons.config') as f:
    polygons_templates = json.load(f)

optimization_weight_function_format = 'polynomial'
if optimization_weight_function_format == 'polynomial':
    configs_file = 'configs/polynomial_movement_optimization.config'
elif optimization_weight_function_format == 'sigmoid':
    configs_file = 'configs/sigmoid_movement_optimization.config'

with open(configs_file, 'r') as f:
    optimization_config = json.load(f)
    n_var = optimization_config['n_variables']
    hyper_parameters = {
        'c1': optimization_config['c1'],
        'c2': optimization_config['c2'],
        'w': optimization_config['w']
    }
    n_particles = optimization_config['n_particles']
    n_iterations = optimization_config['n_iterations']
    n_processes = optimization_config['n_threads']
    max_simulation_steps = optimization_config['max_simulation_steps']
    n_simulated_particles = optimization_config['n_simulated_particles']

    x_min = np.array(
        [
            optimization_config[f"variable_{str(i)}_lower_bound"] for i in range(1, n_var + 1)
        ]
    )
    x_max = np.array(
        [
            optimization_config[f"variable_{str(i)}_upper_bound"] for i in range(1, n_var + 1)
        ]
    )

    # polygon configs
    polygon_set = optimization_config['polygon_set']
    polygons_template = polygons_templates[polygon_set]
    polygons = []
    for polygon_index, polygon_info in polygons_template.items():
        polygon = []
        for corner_index, corner_info in polygon_info.items():
            x = corner_info['x']
            y = corner_info['y']
            polygon.append([x, y])
        polygons.append(np.array(polygon))


def cost_function(
        x,
        max_simulation_steps,
        movement_weight_function,
        n_particles,
        initial_coordinates,
        display_particle_movement=False
):
    costs = np.zeros(x.shape[0])
    penalty_multiplier = 10

    for solution_index in range(x.shape[0]):
        battleground_obj = Battleground(
            calibration_instructions={'movement': True, 'orientation': False},
            weighting_instruction={'movement': True, 'orientation': False},
            polygons=polygons
        )

        particles = []
        for particle_index in range(n_particles):
            particles.append(
                Vector_Player(
                    saving_directory='states/',
                    trainables={'movement': [x[solution_index, var_index] for var_index in range(x.shape[1])]},
                    distance_weight_function=movement_weight_function,
                    velocity=np.array([1, 0]),
                    coordinates=np.array(initial_coordinates[particle_index]),
                    step_size=0.06,
                    n_move_vectors=8,
                    battleground=battleground_obj,
                    inertia_weight=0.3,
                    team='A'
                )
            )

        battleground_obj.set_particles(particles)

        did_collide = False
        for step in range(max_simulation_steps):
            battleground_obj.next(
                plot_properties={
                    'active': True, 'steps': 10, 'fps': 30, 'show_block': False
                } if display_particle_movement else {
                    'active': False
                }
            )
            boundaries_collision, particles_collision = battleground_obj.check_status()

            if boundaries_collision:
                did_collide = True
                break

        movement_areas = battleground_obj.calculate_movement_areas()
        for area in movement_areas:
            costs[solution_index] += 1 - area

        costs[solution_index] += penalty_multiplier * (max_simulation_steps - step) if did_collide else 0

    return costs


if __name__ == '__main__':

    if optimization_weight_function_format == 'polynomial':
        weight_function = WeightFunctions.polynomial_degree_2_distance_weight
    elif optimization_weight_function_format == 'sigmoid':
        weight_function = WeightFunctions.sigmoid_distance_weight

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_var,
        options=hyper_parameters,
        bounds=(x_min, x_max)
    )

    best_cost, best_pos = optimizer.optimize(
        cost_function,
        max_simulation_steps=max_simulation_steps,
        movement_weight_function=weight_function,
        n_particles=n_simulated_particles,
        initial_coordinates=[[0.05, 0.05]],
        iters=n_iterations,
        n_processes=n_processes
    )

    _ = cost_function(
        best_pos.reshape(1, n_var),
        max_simulation_steps=max_simulation_steps,
        movement_weight_function=weight_function,
        n_particles=n_simulated_particles,
        initial_coordinates=[[0.05, 0.05]],
        display_particle_movement=True
    )

    parameters = {
        'parameters': best_pos
    }
    with open('states/movement_algorithm_distance.pickle', 'wb') as f:
        pickle.dump(parameters, f)

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()
