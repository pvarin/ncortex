''' DifferentiableEnv class.
'''
from .base_env import BaseEnv


class DifferentiableEnv(BaseEnv):
    ''' The base class for environments with differentiable dynamics.
    '''

    def __init__(self, dt=0.01):
        self.dt = dt
        super(DifferentiableEnv, self).__init__()

    def dynamics(self, state, action):
        ''' Computes the state derivative.
        '''
        raise NotImplementedError

    def step(self, state, action):
        '''
        Integrates the dynamics using a forward Euler scheme.
        '''
        return state + self.dt * self.dynamics(state, action)
