

class Particle:
    def __init__(self, **kwargs):
        self._velocity = kwargs['velocity'] if 'velocity' in kwargs else None
        self._coordinates = kwargs['coordinates'] if 'coordinates' in kwargs else None
        self._power = kwargs['power'] if 'power' in kwargs else None
        self._step_size = kwargs['step_size'] if 'step_size' in kwargs else None
