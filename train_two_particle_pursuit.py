from classes.vector_player import Vector_Player
from classes.battleground import Battleground
from classes.weight_functions import WeightFunctions
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
import numpy as np
import json
from matplotlib import pyplot as plt
import pickle
from copy import copy

with open('configs/polygons.config') as f:
    polygons_templates = json.load(f)

# Load calibrated movement algorithm
with open('states/movement_algorithm_distance.pickle', 'rb') as f:
    movement_algorithm = pickle.load(f)
    movement_weight_function = WeightFunctions.polynomial_degree_3_distance_weight
    movement_parameters = movement_algorithm['parameters']

optimization_weight_function_format = 'polynomial'
if optimization_weight_function_format == 'polynomial':
    configs_file = 'configs/polynomial_orientation_optimization.config'
elif optimization_weight_function_format == 'sigmoid':
    configs_file = 'configs/sigmoid_orientation_optimization.config'

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
        constants,
        max_simulation_steps,
        calibrated_movement_weight_function,
        calibrated_movement_parameters,
        orientation_weight_function,
        mode,
        display_particle_movement=False
):
    costs = np.zeros(x.shape[0])
    penalty_multiplier = 3

    for solution_index in range(x.shape[0]):
        battleground_obj = Battleground(
            calibration_instructions={'movement': False, 'orientation': True},
            weighting_instruction={'movement': True, 'orientation': True},
            polygons=polygons
        )

        trainables = [x[solution_index, var_index] for var_index in range(x.shape[1])] if mode == 'attack' else constants
        particle_00 = Vector_Player(
            saving_directory='states/',
            trainables={'orientation': trainables},
            non_trainables={'movement': calibrated_movement_parameters},
            distance_weight_function=calibrated_movement_weight_function,
            orientation_weight_function=orientation_weight_function,
            velocity=np.array([1, 0]),
            coordinates=np.array([0.5, 0.1]),
            step_size=0.025,
            n_move_vectors=12,
            battleground=battleground_obj,
            inertia_weight=0.3,
            power=1,
            team='A'
        )
        particle_01 = Vector_Player(
            saving_directory='states/',
            trainables={'orientation': trainables},
            non_trainables={'movement': calibrated_movement_parameters},
            distance_weight_function=calibrated_movement_weight_function,
            orientation_weight_function=orientation_weight_function,
            velocity=np.array([1, 0]),
            coordinates=np.array([0.6, 0.5]),
            step_size=0.025,
            n_move_vectors=12,
            battleground=battleground_obj,
            inertia_weight=0.3,
            power=1,
            team='A'
        )
        particle_02 = Vector_Player(
            saving_directory='states/',
            trainables={'orientation': trainables},
            non_trainables={'movement': calibrated_movement_parameters},
            distance_weight_function=calibrated_movement_weight_function,
            orientation_weight_function=orientation_weight_function,
            velocity=np.array([0, 1]),
            coordinates=np.array([0.1, 0.6]),
            step_size=0.04,
            n_move_vectors=12,
            battleground=battleground_obj,
            inertia_weight=0.3,
            power=1,
            team='A'
        )

        trainables = [x[solution_index, var_index] for var_index in range(x.shape[1])] if mode == 'defense' else constants
        particle_03 = Vector_Player(
            saving_directory='states/',
            trainables={'orientation': trainables},
            non_trainables={'movement': calibrated_movement_parameters},
            distance_weight_function=calibrated_movement_weight_function,
            orientation_weight_function=orientation_weight_function,
            velocity=np.array([-1, 0]),
            coordinates=np.array([0.5, 0.9]),
            step_size=0.06,
            n_move_vectors=12,
            battleground=battleground_obj,
            inertia_weight=0.3,
            power=0.5,
            team='B'
        )

        battleground_obj.set_particles([particle_00, particle_01, particle_02, particle_03])

        did_collide_boundaries = False
        did_collide_each_other = False
        for step in range(1, max_simulation_steps + 1):

            battleground_obj.next(
                plot_properties={
                    'active': True, 'steps': 10, 'fps': 30, 'show_block': False
                } if display_particle_movement else {
                    'active': False
                }
            )

            boundaries_collision, particles_collision = battleground_obj.check_status()

            if boundaries_collision:
                did_collide_boundaries = True
                break

            if particles_collision:
                did_collide_each_other = True
                break

        if mode == 'attack':
            costs[solution_index] = step
        elif mode == 'defense':
            costs[solution_index] = max_simulation_steps - step

        costs[solution_index] += penalty_multiplier * (max_simulation_steps - step) if did_collide_boundaries else 0

    return costs


if __name__ == '__main__':

    if optimization_weight_function_format == 'polynomial':
        weight_function = WeightFunctions.polynomial_degree_3_orientation_weight
    elif optimization_weight_function_format == 'sigmoid':
        weight_function = WeightFunctions.sigmoid_orientation_weight

    phases = ['attack', 'defense']
    n_rotational_training_phases = 20
    constants = np.array([1, 1, 1])
    for train_index in range(1, n_rotational_training_phases):
        mode = phases[train_index % 2]
        print(f'{train_index} => Mode: {mode} - Constants: {constants}')

        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=n_var,
            options=hyper_parameters,
            bounds=(x_min, x_max),
            ftol=0.1,
            ftol_iter=8
        )

        best_cost, best_pos = optimizer.optimize(
            cost_function,
            constants=constants,
            mode=mode,
            max_simulation_steps=max_simulation_steps,
            calibrated_movement_weight_function=movement_weight_function,
            calibrated_movement_parameters=movement_parameters,
            orientation_weight_function=weight_function,
            iters=n_iterations,
            n_processes=n_processes
        )

        _ = cost_function(
            best_pos.reshape(1, n_var),
            constants=constants,
            mode=mode,
            max_simulation_steps=max_simulation_steps,
            calibrated_movement_weight_function=movement_weight_function,
            calibrated_movement_parameters=movement_parameters,
            orientation_weight_function=weight_function,
            display_particle_movement=True
        )

        constants = copy(best_pos)


    # particle.save_trained_state()
    # particle.save_trained_parameters()

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()
