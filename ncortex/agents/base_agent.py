''' BaseAgent class.
'''

class BaseAgent:
    '''Base class for all agents.
    '''

    def __init__(self, env):
        self.env = env

    def act(self):
        ''' Takes a single action on the environment.
        '''
        raise NotImplementedError

    def run(self):
        ''' Acts on the environment until termination.
        '''
        done = False
        while not done:
            _, _, _, done, _ = self.act()
