''' DifferentiableEnv class.
'''
from .base_env import BaseEnv

class DifferentiableEnv(BaseEnv):
    ''' The base class for environments with differentiable dynamics.
    '''

    def __init__(self, x0=None, dt=0.01):
        self.dt = dt
        super(DifferentiableEnv, self).__init__(x0=x0)

    def dynamics(self, action):
        ''' Computes the state derivative.
        '''
        raise NotImplementedError

    def step(self, action):
        '''
        Integrates the dynamics using a forward Euler scheme.
        '''
        return self._state + self.dt * self.dynamics(action)
