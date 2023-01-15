from classes.player import Player
import numpy as np
from copy import copy
from classes.map import Line_With_Vector_And_Point, Line_With_Two_Points
import pickle


class Vector_Player(Player):
    def __init__(self, **kwargs):
        super(Vector_Player, self).__init__(**kwargs)

        # rulers
        self._distance_weight_function = kwargs['distance_weight_function'] if 'distance_weight_function' else None
        self._orientation_weight_function = kwargs['orientation_weight_function'] if 'orientation_weight_function' in kwargs else None
        self._trainable_parameters = kwargs['trainables']
        self._non_trainable_parameters = kwargs['non_trainables'] if 'non_trainables' in kwargs else None

        # general parameters
        self._n_move_vectors = kwargs['n_move_vectors'] if 'n_move_vectors' in kwargs else 12

        # utility parameters
        self._move_vectors = []
        self._distances = []
        self._orientations = []

    def move_one_step(self):
        # Update environmental variables based on instructions
        self.update_environment()

        # Deduct new move vector
        unit_move_vector = self.deduct_new_direction()

        # Store previous coordinates (used for collisions check)
        self._previous_coordinates = copy(self._coordinates)

        # Calculate velocity magnitude
        step_size_multiplier = 1
        if (np.linalg.norm(self._velocity)) != 0:
            direction_change = np.arccos(max(min(np.dot(unit_move_vector, self._velocity) / (np.linalg.norm(unit_move_vector) * np.linalg.norm(self._velocity)), 1), -1))
            if not np.isnan(direction_change):
                step_size_multiplier = 1 - direction_change / np.pi
            else:
                # print('hi')
                pass

        # Update coordinates based on new move vector
        self._coordinates += unit_move_vector * self._step_size * np.sqrt(step_size_multiplier)

        # Store coordinates history
        self._coordinates_history.append(copy(self._coordinates))

        # Update velocity
        self._velocity = copy(unit_move_vector)

    def update_environment(self):
        self.construct_move_vectors()

        if self._battleground.get_weighting_instructions()['movement']:
            self.find_distance_to_boundaries()
        if self._battleground.get_weighting_instructions()['orientation']:
            self.find_disorientation_to_target()

    def find_disorientation_to_target(self):
        self._orientations.clear()
        target_vector = self._target_particle.get_particle().get_coordinates() - self._coordinates

        for move_vector in self._move_vectors:
            self._orientations.append(
                np.arccos(np.dot(move_vector, target_vector) / (np.linalg.norm(move_vector) * np.linalg.norm(target_vector)))
            )

    def deduct_new_direction(self):

        new_move_vector = np.zeros(2)
        for i in range(len(self._move_vectors)):

            # Apply movement general instructions
            move_vector_weight = 1
            if self._battleground.get_weighting_instructions()['movement']:
                distance = self._distances[i]
                if self._battleground.get_calibration_instructions()['movement']:
                    distance_weight = self._distance_weight_function(
                        distance,
                        self._trainable_parameters['movement']
                    )
                else:
                    distance_weight = self._distance_weight_function(
                        distance,
                        self._non_trainable_parameters['movement']
                    )

                move_vector_weight *= distance_weight

            # Apply pursuit instructions
            if self._battleground.get_weighting_instructions()['orientation']:
                orientation = self._orientations[i]
                if self._battleground.get_calibration_instructions()['orientation']:
                    orientation_weight = self._orientation_weight_function(orientation, self._trainable_parameters['orientation'], self._target_particle.get_plan())
                else:
                    orientation_weight = self._orientation_weight_function(orientation, self._non_trainable_parameters['orientation'], self._target_particle.get_plan())
                move_vector_weight *= orientation_weight

            # update new_move_vector according to this move_vector
            new_move_vector += move_vector_weight * np.array(self._move_vectors[i])

        # Calculate unit move vector
        if (np.linalg.norm(new_move_vector)) != 0:
            unit_move_vector = new_move_vector / (np.linalg.norm(new_move_vector))
        else:
            unit_move_vector = new_move_vector

        # Apply inertia
        merged_vector = self._inertia_weight * self._velocity + (1-self._inertia_weight) * unit_move_vector

        # Calculate final unit vector
        if (np.linalg.norm(merged_vector)) != 0:
            unit_merged_vector = merged_vector / (np.linalg.norm(merged_vector))
        else:
            unit_merged_vector = merged_vector

        return unit_merged_vector

    def construct_move_vectors(self):
        base_vector = self._velocity

        theta_array = np.linspace(-np.pi, np.pi, self._n_move_vectors)

        self._move_vectors.clear()
        for theta_index in range(theta_array.shape[0]):
            theta = theta_array[theta_index]
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]
            )

            self._move_vectors.append(np.matmul(rotation_matrix, base_vector))

    def find_distance_to_boundaries(self):
        self._distances.clear()

        for move_vector in self._move_vectors:
            distances = np.zeros(len(self._battleground.get_map_boundaries()))
            counter = 0
            for boundary in self._battleground.get_map_boundaries():
                distance = boundary.find_distance_to_line(
                    Line_With_Vector_And_Point(
                        vector=move_vector,
                        point=self._coordinates
                    )
                )

                distances[counter] = distance if distance is not None else np.inf
                counter += 1

            self._distances.append(np.min(distances))

    # Save trained state
    def save_trained_state(self):
        with open(f'{self._saving_directory}_particle.pickle', 'wb') as f:
            pickle.dump(self, f)

    def save_trained_parameters(self):
        if self._battleground.get_weighting_instructions()['movement']:
            if self._battleground.get_calibration_instructions()['movement']:
                with open(f'{self._saving_directory}movement_trained_parameters.pickle', 'wb') as f:
                    pickle.dump(self._trainable_parameters['movement'], f)

        if self._battleground.get_weighting_instructions()['orientation']:
            if self._battleground.get_calibration_instructions()['orientation']:
                with open(f'{self._saving_directory}orientation_trained_parameters.pickle', 'wb') as f:
                    pickle.dump(self._trainable_parameters['orientation'], f)

    @staticmethod
    def load_trained_particle(loading_directory):
        try:
            with open(loading_directory, 'rb') as f:
                loaded_object = pickle.load(f)
            return loaded_object
        except Exception as e:
            print(f'Loading particle crashed with error: {e}')
            exit()

    @staticmethod
    def load_trained_parameters(loading_directory):
        try:
            with open(loading_directory, 'rb') as f:
                loaded_parameters = pickle.load(f)
            return loaded_parameters
        except Exception as e:
            print(f'Loading particle crashed with error: {e}')
            exit()
