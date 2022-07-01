from classes.map import Map
import numpy as np

map = Map(
    polygons=
    [
        np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]]
        )
    ]
)
map.initialize_particles()

print('hi')