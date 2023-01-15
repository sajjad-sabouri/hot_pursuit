import numpy as np


class WeightFunctions:
    @staticmethod
    def sigmoid_distance_weight(distance, parameters):
        return 1 / (1 + np.exp(-1 * parameters[0] * (distance - parameters[1])))

    @staticmethod
    def polynomial_degree_2_distance_weight(distance, parameters):
        return parameters[0] * np.power(distance, 2) + parameters[1] * distance

    @staticmethod
    def polynomial_degree_3_distance_weight(distance, parameters):
        return parameters[0] * np.power(distance, 3) + parameters[1] * np.power(distance, 2) + parameters[2] * distance

    @staticmethod
    def sigmoid_orientation_weight(angle, parameters, plan):
        if plan == 'attack':
            core = np.pi - angle
        elif plan == 'defense':
            core = angle
        return 1 / (1 + np.exp(-1 * parameters[0] * (core - parameters[1])))

    @staticmethod
    def polynomial_degree_2_orientation_weight(angle, parameters, plan):
        if plan == 'attack':
            core = np.pi - angle
        elif plan == 'defense':
            core = angle
        return parameters[0] * np.power(core, 2) + parameters[1] * core

    @staticmethod
    def polynomial_degree_3_orientation_weight(angle, parameters, plan):
        if plan == 'attack':
            core = np.pi - angle
        elif plan == 'defense':
            core = angle
        return parameters[0] * np.power(core, 3) + parameters[1] * np.power(core, 2) + parameters[2] * core
