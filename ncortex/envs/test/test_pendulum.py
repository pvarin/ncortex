''' Test the Pendulum class.
'''

import tensorflow as tf
import numpy as np
from ncortex.envs import Pendulum


class TestPendulum(tf.test.TestCase):
    ''' TestCase for the Pendulum class and methods.
    '''

    def setUp(self):
        ''' Initializes the test case with a Pendulum object.
        '''
        self.Q = 3. * np.eye(2, dtype=np.float32)
        self.R = 4. * np.eye(1, dtype=np.float32)
        self.pend = Pendulum(Q=self.Q, R=self.R)
        self.pend.reset()

    def test_transition_cost(self):
        ''' Test the transition_cost function.
        '''
        with self.cached_session():
            # Compute the transition cost with an arbitrary state and action.
            state = tf.constant([1., 2.])
            action = tf.constant([3.])
            cost = self.pend.transition_cost(state, action)

            # Test the value.
            self.assertAllClose(cost, self.Q[0, 0] * (1. + 4.) + self.R[0, 0] * 9.)

            # Test the gradients.
            grad = tf.gradients(cost, [state, action])
            self.assertAllClose(grad[0], 2. * self.Q[0, 0] * state)
            self.assertAllClose(grad[1], 2. * self.R[0, 0] * action)

    def test_dynamics(self):
        ''' Test the step function.
        '''
        with self.cached_session():
            # Compute dynamics with an aribtrary state and action.
            state = tf.constant([1., 2.])
            self.pend.state = state
            action = tf.constant([1.])
            deriv = self.pend.dynamics(action)

            # Test the shape of the dynamics
            self.assertEqual(state.shape, deriv.shape)

            # Test that the value of the dynamics
            self.assertAllClose(deriv[0], state[1])
            self.assertAllClose(deriv[1], -tf.sin(state[0]) + action[0])


if __name__ == '__main__':
    tf.test.main()
