''' RandomAgent class.
'''

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    '''
    A random agent that samples from the environment action space and takes a
    step.
    '''
    def act(self):
        action = self.env.action_space.sample()
        return self.env.step(action)
