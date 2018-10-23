''' Test the Pendulum class.
'''
import unittest
import tensorflow as tf
from ncortex.envs import Pendulum


class TestPendulum(unittest.TestCase):
    ''' TestCase for the Pendulum class and methods.
    '''

    def setUp(self):
        ''' Initializes the test case with a Pendulum object.
        '''
        self.pend = Pendulum()

    def test_transition_cost(self):
        ''' Test the transition_cost function.
        '''
        pass
        # self.pend.transition_cost()

    def test_step(self):
        ''' Test the step function.
        '''
        state = tf.constant([0.0, 0.0])
        action = tf.constant(1.0)
        self.pend.step(state, action)


if __name__ == '__main__':
    unittest.main()
