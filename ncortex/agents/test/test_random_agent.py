''' Test the RandomAgent class.
'''

import tensorflow as tf
from ncortex.envs import Pendulum
from ncortex.envs import StatefulEnv
from ncortex.agents import RandomAgent

class TestRandomAgent(tf.test.TestCase):
    ''' Test the RandomAgent class.
    '''
    def setUp(self):
        ''' Set up the test case with a RandomAgent on a Pendulum environment.
        '''
        stateless_env = Pendulum()
        self.env = StatefulEnv(stateless_env)
        self.env.reset()
        self.agent = RandomAgent(self.env)

    def test_act(self):
        ''' Test the act method.
        '''
        with self.cached_session():
            self.agent.act()

if __name__ == '__main__':
    tf.test.main()
