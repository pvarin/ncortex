''' BaseEnv class.
'''


class BaseEnv:
    ''' The base class for all environments.
    '''
    def __init__(self, x0=None):
        if x0 is None:
            self.reset()
        else:
            self._state = x0

    def transition_cost(self, state, action):
        ''' The cost of being in a state and taking an action.
        '''
        raise NotImplementedError

    def final_cost(self, state):
        ''' The cost of ending the simulation in a particular state.
        '''
        raise NotImplementedError

    @property
    def action_space(self):
        ''' The space a random agent, should sample from.
        '''
        raise NotImplementedError

    def step(self, action):
        '''
        Takes an action in a particular state and returns the next state. Uses
        a forward Euler integration scheme.
        '''
        raise NotImplementedError

    def reset(self):
        ''' Resets the environment state.
        '''
        raise NotImplementedError
