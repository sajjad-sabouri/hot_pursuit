from classes.free_player import Free_Player
from classes.battleground import Battleground
from classes.weight_functions import WeightFunctions
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_cost_history
import numpy as np
import json
from matplotlib import pyplot as plt
import pickle
from classes.gravity_driver import Gravity_Driver
from copy import copy


with open('configs/polygons.config') as f:
    polygons_templates = json.load(f)


optimization_weight_function_format = 'rbf'
if optimization_weight_function_format == 'rbf':
    configs_file = 'configs/rbf_movement_optimization.config'


with open(configs_file) as f:
    optimization_config = json.load(f)

    # opt configs
    hyper_parameters = {
        'c1': optimization_config['c1'],
        'c2': optimization_config['c2'],
        'w': optimization_config['w']
    }
    n_particles = optimization_config['n_particles']
    n_iterations = optimization_config['n_iterations']
    n_processes = optimization_config['n_threads']

    # problem configs
    n_var = optimization_config['n_variables']
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
        n_particles,
        initial_coordinates,
        drivers,
        polygons,
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

        to_be_used_drivers = copy(drivers)

        to_be_used_drivers.extend(
            [
                Gravity_Driver(
                    gravity_center=np.array([x[solution_index, 0], x[solution_index, 1]]),
                    gravity_intensity_center=np.array([x[solution_index, 2], x[solution_index, 3]]),
                    gravity_intensity_variance=np.array([x[solution_index, 4], x[solution_index, 5]])
                )
            ]
        )

        particles = []
        for particle_index in range(n_particles):
            particles.append(
                Free_Player(
                    saving_directory='states/',
                    drivers=to_be_used_drivers,
                    velocity=np.array([1, 0]),
                    coordinates=np.array(initial_coordinates[particle_index]),
                    step_size=0.05,
                    battleground=battleground_obj
                )
            )

        battleground_obj.set_particles(particles)

        did_collide = False
        for step in range(max_simulation_steps):
            battleground_obj.next(
                plot_properties={
                    'active': True, 'steps': 3, 'fps': 60, 'show_block': False
                } if display_particle_movement else {
                    'active': False
                }
            )
            boundaries_collision, particles_collision = battleground_obj.check_status()

            if boundaries_collision:
                did_collide = True
                break

        # objective func.
        movement_areas = battleground_obj.calculate_movement_areas()
        for area in movement_areas:
            costs[solution_index] += 1 - area

        costs[solution_index] += penalty_multiplier * (max_simulation_steps - step) if did_collide else 0

    return costs


if __name__ == '__main__':


    # define optimizer
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_var,
        options=hyper_parameters,
        bounds=(x_min, x_max)
    )

    # initial drivers
    drivers = [
        Gravity_Driver(
            gravity_center=np.array([0.9, 0.9]),
            gravity_intensity_center=np.array([0.9, 0.05]),
            gravity_intensity_variance=np.array([0.1, 0.1])
        ),
        Gravity_Driver(
            gravity_center=np.array([0.9, 0.05]),
            gravity_intensity_center=np.array([0.05, 0.05]),
            gravity_intensity_variance=np.array([0.1, 0.1])
        )
    ]

    # run suboptimal processes
    n_suboptimal_iterations = 10
    for suboptimal_iter in range(n_suboptimal_iterations):

        print(f'Processing new suboptimal solution iteration {suboptimal_iter} ...')
        best_cost, best_pos = optimizer.optimize(
            cost_function,
            max_simulation_steps=max_simulation_steps,
            n_particles=n_simulated_particles,
            initial_coordinates=[[0.05, 0.05], [0.9, 0.9], [0.7, 0.6], [0.2, 0.5]],
            drivers=drivers,
            polygons=polygons,
            iters=n_iterations,
            n_processes=n_processes
        )

        if suboptimal_iter % 2 == 0:
            _ = cost_function(
                best_pos.reshape(1, n_var),
                max_simulation_steps=max_simulation_steps,
                n_particles=n_simulated_particles,
                initial_coordinates=[[0.05, 0.05], [0.9, 0.9], [0.7, 0.6], [0.2, 0.5]],
                drivers=drivers,
                polygons=polygons,
                display_particle_movement=True
            )

        drivers.append(
            Gravity_Driver(
                gravity_center=np.array([best_pos[0], best_pos[1]]),
                gravity_intensity_center=np.array([best_pos[2], best_pos[3]]),
                gravity_intensity_variance=np.array([best_pos[4], best_pos[5]]),
            )
        )

        with open('states/movement_algorithm_rbf_test.pickle', 'wb') as f:
            pickle.dump(drivers, f)

    # animate movement for best solution found
    _ = cost_function(
        best_pos.reshape(1, n_var),
        max_simulation_steps=max_simulation_steps,
        n_particles=n_simulated_particles,
        initial_coordinates=[[0.05, 0.05], [0.9, 0.9], [0.7, 0.6], [0.2, 0.5]],
        drivers=drivers,
        polygons=polygons,
        display_particle_movement=True
    )

    with open('states/movement_algorithm_rbf.pickle', 'wb') as f:
        pickle.dump(drivers, f)
