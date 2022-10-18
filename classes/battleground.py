from classes.map import Map_Boundaries
from classes.target import Target
import numpy as np
from matplotlib import pyplot as plt


class Battleground:
    def __init__(self, **kwargs):
        # rulers
        self._calibration_instructions = kwargs['calibration_instructions']
        self._weighting_instructions = kwargs['weighting_instruction']

        # general parameters
        self._map_boundaries = Map_Boundaries(
            polygons=kwargs['polygons'] if 'polygons' in kwargs else None
        )

        # utility parameters
        self._steps = 0
        self._particles = []

    def update_targets(self):
        # Define particles targets
        for particle in self._particles:
            for other_particle in self._particles:
                if particle.get_team() != other_particle.get_team():
                    if particle != other_particle:
                        if particle.get_power() > other_particle.get_power():
                            particle.set_target(
                                Target(
                                    particle=other_particle,
                                    plan='attack'
                                )
                            )
                        else:
                            particle.set_target(
                                Target(
                                    particle=other_particle,
                                    plan='defense'
                                )
                            )

    def next(self, plot_properties={'active': False, 'steps': 1, 'fps': 0.05, 'show_block': False}):

        self._steps += 1

        if self._weighting_instructions['orientation']:
            self.update_targets()

        for particle in self._particles:
            particle.move_one_step()

        if plot_properties['active']:
            self.clear_canvas()
            self.draw_battle_map()
            self.draw_battle_particles(plot_properties['steps'])
            # self.draw_gravity_centers()
            plt.title(f'Time Index {str(self._steps)}')
            plt.show(block=plot_properties['show_block'])
            plt.pause(1 / plot_properties['fps'])

    def calculate_movement_areas(self):
        movement_areas = []

        for particle in self._particles:
            movement_areas.append(particle.calculate_movement_area())

        return movement_areas

    def check_status(self):
        did_particles_collide_with_boundaries = False
        did_particles_collide_with_each_other = False

        if self._weighting_instructions['movement']:
            did_particles_collide_with_boundaries = self.check_particles_collision_with_boundaries()

        if self._weighting_instructions['orientation']:
            did_particles_collide_with_each_other = self.check_particles_collisions_with_each_other()

        return did_particles_collide_with_boundaries, did_particles_collide_with_each_other

    def check_particles_collisions_with_each_other(self):
        for particle in self._particles:
            for other_particle in self._particles:
                if particle.get_team() != other_particle.get_team():
                    if particle != other_particle:
                        if np.linalg.norm(particle.get_coordinates() - other_particle.get_coordinates()) < 0.05:
                            return True
        return False

    def check_particles_collision_with_boundaries(self):
        for particle in self._particles:
            if particle.did_particle_collide_with_boundaries():
                return True
        return False

    # Drawers
    def draw_battle_map(self):
        self._map_boundaries.draw_boundaries()

    def draw_battle_particles(self, steps=1):

        for i in range(len(self._particles)):
            movement_history = self._particles[i].get_movement_history()
            steps = min(steps, len(movement_history))
            for step in range(1, steps + 1):
                plt.scatter(
                    movement_history[-step][0],
                    movement_history[-step][1],
                    color='crimson' if self._particles[i].get_team() == 'A' else 'royalblue',
                    alpha=np.power((1 / step), 2)
                )

    def draw_gravity_centers(self):

        for i in range(len(self._particles)):
            gravity_centers = self._particles[i].get_gravity_centers_linked()
            for j in range(len(gravity_centers)):
                plt.scatter(
                    gravity_centers[j][0][0],
                    gravity_centers[j][0][1],
                    color='royalblue',
                    alpha=0.3,
                    marker='o'
                )
                plt.scatter(
                    gravity_centers[j][1][0],
                    gravity_centers[j][1][1],
                    color='tomato',
                    alpha=0.3,
                    marker='.'
                )
                plt.plot(
                    [gravity_centers[j][0][0], gravity_centers[j][1][0]],
                    [gravity_centers[j][0][1], gravity_centers[j][1][1]],
                    color='silver',
                    alpha=0.2,
                    linewidth=1.5
                )

    def clear_canvas(self):
        plt.cla()
        plt.clf()

    # Getters and setters
    def get_map_boundaries(self):
        return self._map_boundaries.get_boundaries()

    def get_map_corners(self):
        return self._map_boundaries.get_map_min_max_corners()

    def get_weighting_instructions(self):
        return self._weighting_instructions

    def get_calibration_instructions(self):
        return self._calibration_instructions

    def set_particles(self, particles):
        self._particles = particles
