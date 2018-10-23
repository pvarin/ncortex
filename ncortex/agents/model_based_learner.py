''' ModelBasedLearner class.
'''
from .actor_base import ActorBase


class ModelBasedLearner(ActorBase):
    '''
    Base class for all model-based reinforcement learning actors. The model
    can be learned or specified in advance.
    '''

    def __init__(self, env):
        self.env = env
        super(ModelBasedLearner, self).__init__()

    def act(self):
        ''' Takes a single action on the environment.
        '''
        action = None
        return self.env.step(action)
