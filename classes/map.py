import numpy as np
from matplotlib import pyplot as plt


class Line_With_Vector_And_Point:
    def __init__(self, **kwargs):
        self._point = kwargs['point'] if 'point' in kwargs else None
        self._vector = kwargs['vector'] if 'vector' in kwargs else None
        self.find_valid_min_and_max()
        self.find_line_properties()

    def find_valid_min_and_max(self):
        if self._vector[0] > 0:
            self._x_min = self._point[0]
            self._x_max = np.inf
        else:
            self._x_min = -1 * np.inf
            self._x_max = self._point[0]

        if self._vector[1] > 0:
            self._y_min = self._point[1]
            self._y_max = np.inf
        else:
            self._y_min = -1 * np.inf
            self._y_max = self._point[1]

    def find_line_properties(self):
        if self._vector[0] != 0:
            self._slope = self._vector[1] / self._vector[0]
            self._intercept = self._point[1] - self._slope * self._point[0]
        else:
            self._slope = np.inf
            self._intercept = np.inf

    def get_slope(self):
        return self._slope

    def get_intercept(self):
        return self._intercept

    def get_point(self):
        return self._point


class Line_With_Two_Points:
    def __init__(self, **kwargs):
        self._point_one = kwargs['point_one'] if 'point_one' in kwargs else None
        self._point_two = kwargs['point_two'] if 'point_two' in kwargs else None
        self.find_valid_min_and_max()
        self.calculate_line_properties()

    def find_valid_min_and_max(self):
        self._x_min = np.min(np.array([self._point_one[0], self._point_two[0]]))
        self._x_max = np.max(np.array([self._point_one[0], self._point_two[0]]))
        self._y_min = np.min(np.array([self._point_one[1], self._point_two[1]]))
        self._y_max = np.max(np.array([self._point_one[1], self._point_two[1]]))

    def calculate_line_properties(self):
        if self._point_one[0] != self._point_two[0]:
            self._slope = (self._point_one[1] - self._point_two[1]) / (self._point_one[0] - self._point_two[0])
            self._intercept = self._point_one[1] - self._slope * self._point_one[0]
        else:
            self._slope = np.inf
            self._intercept = np.inf

    def find_distance_to_line(self, line):

        if self._slope == line.get_slope():
            return np.inf

        if self._intercept == np.inf:
            x_intersection = self._point_one[0]
            y_intersection = line.get_slope() * x_intersection + line.get_intercept()

        elif line.get_intercept() == np.inf:
            x_intersection = line.get_point()[0]
            y_intersection = self._slope * x_intersection + self._intercept

        else:
            x_intersection = (line.get_intercept() - self._intercept) / (self._slope - line.get_slope())
            y_intersection = self._slope * x_intersection + self._intercept

        if (self._x_max >= x_intersection >= self._x_min) and (line._x_max >= x_intersection >= line._x_min):
            if (self._y_max >= y_intersection >= self._y_min) and (line._y_max >= y_intersection >= line._y_min):
                return np.sqrt(
                    np.power(line.get_point()[0] - x_intersection, 2) + np.power(line.get_point()[1] - y_intersection, 2)
                )

        return np.inf

    def do_intersect(self, line):
        dist = self.find_distance_to_line(line)
        if dist < line.get_length():
            return True
        return False

    def get_slope(self):
        return self._slope

    def get_intercept(self):
        return self._intercept

    def get_length(self):
        return np.linalg.norm(self._point_one - self._point_two)

    def get_point(self):
        return self._point_one


class Map_Boundaries:
    def __init__(self, **kwargs):
        self._polygons = kwargs['polygons'] if 'polygons' in kwargs else None
        self._boundaries = []
        self.extract_boundaries()

    def extract_boundaries(self):
        for polygon in self._polygons:
            for corner_index in range(polygon.shape[0]):

                corner_one = polygon[corner_index, :]

                if corner_index == polygon.shape[0] - 1:
                    corner_two = polygon[0, :]
                else:
                    corner_two = polygon[corner_index + 1, :]

                self._boundaries.append(
                    Line_With_Two_Points(
                        point_one=corner_one,
                        point_two=corner_two
                    )
                )

    def get_boundaries(self):
        return self._boundaries

    def draw_boundaries(self):
        for polygon in self._polygons:
            plt.plot(
                polygon[:, 0],
                polygon[:, 1],
                'r-'
            )
            plt.plot(
                [polygon[0, 0], polygon[-1, 0]],
                [polygon[0, 1], polygon[-1, 1]],
                'r-'
            )

    def get_map_min_max_corners(self):

        min_x, min_y = np.inf, np.inf
        max_x, max_y = -1 * np.inf, -1 * np.inf

        for boundary in self._boundaries:
            if boundary._x_min < min_x:
                min_x = boundary._x_min
            if boundary._x_max > max_x:
                max_x = boundary._x_max
            if boundary._y_min < min_y:
                min_y = boundary._y_min
            if boundary._y_max > max_y:
                max_y = boundary._y_max

        return min_x, max_x, min_y, max_y