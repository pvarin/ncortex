''' RandomAgent class.
'''

from .agent_base import AgentBase


class RandomAgent(AgentBase):
    '''
    A random agent that samples from the environment action space and takes a
    step.
    '''
    def act(self):
        action = self.env.action_space.sample()
        return self.env.step(action)
