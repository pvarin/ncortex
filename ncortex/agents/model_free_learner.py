''' ModelBasedLearner class.
'''
from .actor_base import ActorBase


class ModelFreeLearner(ActorBase):
    '''
    Base class for all model-based reinforcement learning actors. The model
    can be learned or specified in advance.
    '''

    def __init__(self, env):
        self.env = env
        super(ModelFreeLearner, self).__init__()

    def act(self):
        ''' Takes a single action on the environment.
        '''
        # TODO: implement this
        pass
