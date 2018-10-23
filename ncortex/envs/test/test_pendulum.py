''' Test the Pendulum class.
'''

import tensorflow as tf
from ncortex.envs import Pendulum


class TestPendulum(tf.test.TestCase):
    ''' TestCase for the Pendulum class and methods.
    '''

    def setUp(self):
        ''' Initializes the test case with a Pendulum object.
        '''
        self.pend = Pendulum()

    def test_transition_cost(self):
        ''' Test the transition_cost function.
        '''
        with self.cached_session():
            state = tf.constant([[1.0], [2.0]])
            action = tf.constant([[3.0]])
            self.pend.transition_cost(state, action)

    def test_step(self):
        ''' Test the step function.
        '''
        with self.cached_session():
            state = tf.constant([[0.0, 0.0]])
            self.pend.set_state(state)
            action = tf.constant([[1.0]])
            self.pend.step(action)


if __name__ == '__main__':
    tf.test.main()
