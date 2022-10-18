

class Target:
    def __init__(self, **kwargs):
        self._particle = kwargs['particle'] if 'particle' in kwargs else None
        self._plan = kwargs['plan'] if 'plan' in kwargs else None

    def get_particle(self):
        return self._particle

    def get_plan(self):
        return self._plan

