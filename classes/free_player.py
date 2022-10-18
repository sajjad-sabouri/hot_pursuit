from classes.player import Player
import numpy as np
from copy import copy
from classes.map import Line_With_Vector_And_Point, Line_With_Two_Points


class Free_Player(Player):
    def __init__(self, **kwargs):
        super(Free_Player, self).__init__(**kwargs)

        # rulers
        self._drivers = kwargs['drivers']
        self._orientation_weight_function = kwargs['orientation_weight_function'] if 'orientation_weight_function' in kwargs else None
        self._trainable_parameters = kwargs['trainables'] if 'trainables' in kwargs else None

        # general parameters

        # utility parameters
        self._intensities = []
        self._unit_move_vectors = []
        self._orientations = []

    def move_one_step(self):

        # Update environment variables
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
        self._unit_move_vectors.clear()
        self._intensities.clear()
        self._orientations.clear()

        if self._battleground.get_weighting_instructions()['orientation']:
            target_vector = self._target_particle.get_particle().get_coordinates() - self._coordinates

        for driver in self._drivers:
            unit_move_vector, intensity = driver.calculate_driver_gravity(self._coordinates)

            self._unit_move_vectors.append(unit_move_vector)
            self._intensities.append(intensity)
            if self._battleground.get_weighting_instructions()['orientation']:
                self._orientations.append(
                    np.arccos(np.dot(unit_move_vector, target_vector) / (np.linalg.norm(unit_move_vector) * np.linalg.norm(target_vector)))
                )

    def deduct_new_direction(self):

        new_move_vector = np.zeros(2)

        # Apply movement general instructions
        if self._battleground.get_weighting_instructions()['movement']:
            for i in range(len(self._unit_move_vectors)):
                disorientation = np.arccos(
                    max(min(np.dot(self._velocity, self._unit_move_vectors[i]) / (np.linalg.norm(self._unit_move_vectors[i]) * np.linalg.norm(self._velocity)), 1), -1)
                )
                disorientation_penalty = np.sqrt(1-disorientation/np.pi)
                if not np.isnan(disorientation):
                    new_move_vector += self._intensities[i] * self._unit_move_vectors[i] * disorientation_penalty
                else:
                    new_move_vector += self._intensities[i] * self._unit_move_vectors[i]

                if self._battleground.get_weighting_instructions()['orientation']:
                    orientation = self._orientations[i]
                    orientation_weight = self._orientation_weight_function(orientation, self._trainable_parameters['orientation'], self._target_particle.get_plan())
                    new_move_vector *= orientation_weight

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

    # Getters and setters
    def get_gravity_centers(self):
        return [driver.get_gravity_center() for driver in self._drivers]

    def get_gravity_centers_linked(self):
        return [driver.get_gravity_centers() for driver in self._drivers]
