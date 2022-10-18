from classes.map import Line_With_Vector_And_Point, Line_With_Two_Points
import numpy as np
from copy import copy
import pickle
import pandas as pd


class Player:
    def __init__(self, **kwargs):

        # instantaneous parameters
        self._velocity = kwargs['velocity'] if 'velocity' in kwargs else np.zeros(2)
        self._coordinates = kwargs['coordinates'] if 'coordinates' in kwargs else np.zeros(2)
        self._power = kwargs['power'] if 'power' in kwargs else 0
        self._team = kwargs['team']

        # general parameters
        self._step_size = kwargs['step_size'] if 'step_size' in kwargs else 1
        self._battleground = kwargs['battleground'] if 'battleground' in kwargs else None
        self._inertia_weight = kwargs['inertia_weight'] if 'inertia_weight' in kwargs else 0
        self._saving_directory = kwargs['saving_directory']

        # utility parameters
        self._coordinates_history = [copy(self._coordinates)]
        self._previous_coordinates = None
        self._target_particle = None

    def did_particle_collide_with_boundaries(self):
        for boundary in self._battleground.get_map_boundaries():
            if boundary.do_intersect(
                Line_With_Two_Points(
                    point_one=self._previous_coordinates,
                    point_two=self._coordinates
                )
            ):
                return True
        return False

    def calculate_wrapper_movement_area(self):
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -1 * np.inf, -1 * np.inf

        for coordinates in self._coordinates_history:
            if coordinates[0] < min_x:
                min_x = coordinates[0]
            if coordinates[0] > max_x:
                max_x = coordinates[0]
            if coordinates[1] < min_y:
                min_y = coordinates[1]
            if coordinates[1] > max_y:
                max_y = coordinates[1]

        return (max_x - min_x) * (max_y - min_y)

    def calculate_movement_area(self):
        min_x, max_x, min_y, max_y = self._battleground.get_map_corners()
        mesh_size = 8
        x_mesh = np.linspace(min_x, max_x, mesh_size)
        y_mesh = np.linspace(min_y, max_y, mesh_size)

        traversed_grids = []
        for coordinate in self._coordinates_history:
            found = False

            for i in range(1, x_mesh.shape[0]):

                for j in range(1, y_mesh.shape[0]):

                    if x_mesh[i-1] <= coordinate[0] <= x_mesh[i] and y_mesh[j-1] <= coordinate[1] <= y_mesh[j]:
                        traversed_grids.append([i, j])
                        found = True
                        break

                if found:
                    break

        df_traversed = pd.DataFrame(traversed_grids)
        df_traversed.drop_duplicates(inplace=True)
        return df_traversed.shape[0]/mesh_size**2

    # Getters and setters
    def set_distance_weights(self, alpha, beta):
        self._distance_weight_alpha = alpha
        self._distance_weight_beta = beta

    def set_orientation_weights(self, alpha, beta):
        self._orientation_weight_alpha = alpha
        self._orientation_weight_beta = beta

    def get_power(self):
        return self._power

    def get_coordinates(self):
        return self._coordinates

    def set_target(self, target):
        self._target_particle = target
        # self.find_disorientation_to_target()

    def get_movement_history(self):
        return self._coordinates_history

    def get_team(self):
        return self._team
