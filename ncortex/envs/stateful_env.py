''' StatefulEnv class.
'''


class StatefulEnv:
    ''' The base class for all environments.
    '''

    def __init__(self, env):
        self.env = env
        self.state = env.x_0

    def step(self, action):
        '''
        Takes an action in a particular state and returns the next state. Uses
        a forward Euler integration scheme.
        '''
        return self.env.step(self.state, action)

    @property
    def action_space(self):
        ''' Forward the underlying environment's action space.
        '''
        return self.env.action_space

    @property
    def observation_space(self):
        ''' Forward the underlying environment's observation space.
        '''
        return self.env.observation_space

    def render(self):
        '''
        Takes an action in a particular state and returns the next state. Uses
        a forward Euler integration scheme.
        '''
        return self.env.render(self.state)

    def reset(self):
        ''' Resets the environment state.
        '''
        self.state = self.env.reset()
