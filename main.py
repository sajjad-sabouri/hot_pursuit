from classes.battleground import Battleground
import numpy as np

battle = Battleground(
    polygons=
    [
        np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]]
        ),
        np.array(
            [[0.8, 0.1], [0.9, 0.1], [0.9, 0.8], [0.8, 0.8]]
        )
    ]
)

battle.initialize_sample_particles()

for i in range(100):
    battle.next()

print('Game is finished!')
