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
        self.x0 = tf.constant([1., 2.])
        self.Q = 3. * np.eye(2, dtype=np.float32)
        self.Q_f = 4. * np.eye(2, dtype=np.float32)
        self.R = 5. * np.eye(1, dtype=np.float32)
        self.pend = Pendulum(x0=self.x0, Q=self.Q, Q_f=self.Q_f, R=self.R)
        self.pend.reset()

    def test_transition_cost(self):
        ''' Test the transition_cost method.
        '''
        with self.cached_session():
            # Compute the transition cost with an arbitrary state and action.
            state = tf.constant([1., 2.])
            action = tf.constant([3.])
            cost = self.pend.transition_cost(state, action)

            # Test the value.
            self.assertAllClose(cost,
                                self.Q[0, 0] * (1. + 4.) + self.R[0, 0] * 9.)

            # Test the gradients.
            grad = tf.gradients(cost, [state, action])
            self.assertAllClose(grad[0], 2. * self.Q[0, 0] * state)
            self.assertAllClose(grad[1], 2. * self.R[0, 0] * action)

    def test_vectorized_transition_cost(self):
        ''' Test the transition_cost method with a vectorized input.
        '''
        with self.cached_session():
            # Compute the transition cost with an arbitrary state and action.
            state = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
            action = tf.constant([[1.], [2.], [3.]])
            cost = self.pend.transition_cost(state, action)

            # Test the value.
            self.assertAllClose(
                cost, self.Q[0, 0] * tf.constant([5., 25., 61.]) +
                self.R[0, 0] * tf.constant([1., 4., 9.]))

            # Test the gradients.
            grad = tf.gradients(cost, [state, action])
            self.assertAllClose(grad[0], 2. * self.Q[0, 0] * state)
            self.assertAllClose(grad[1], 2. * self.R[0, 0] * action)

    def test_final_cost(self):
        ''' Test the final cost method.
        '''
        with self.cached_session():
            # Compute the transition cost with an arbitrary state and action.
            state = tf.constant([1., 2.])
            cost = self.pend.final_cost(state)

            # Test the value.
            self.assertAllClose(cost, self.Q_f[0, 0] * (1. + 4.))

            # Test the gradients.
            grad = tf.gradients(cost, [state])
            self.assertAllClose(grad[0], 2. * self.Q_f[0, 0] * state)

    def test_vectorized_final_cost(self):
        ''' Test the final cost method with a vectorized state.
        '''
        with self.cached_session():
            # Compute the final cost at an arbitrary state.
            state = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
            cost = self.pend.final_cost(state)

            # Test the value.
            self.assertAllClose(cost,
                                self.Q_f[0, 0] * tf.constant([5., 25., 61.]))

            # Test the gradients.
            grad = tf.gradients(cost, [state])
            self.assertAllClose(grad[0], 2. * self.Q_f[0, 0] * state)

    def test_dynamics(self):
        ''' Test the dynamics method.
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

    def test_vectorized_dynamics(self):
        ''' Test the dynamics method with a vectorized input.
        '''
        with self.cached_session():
            # Compute dynamics with an aribtrary state and action.
            state = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
            self.pend.state = state
            action = tf.constant([[1.], [2.], [3.]])
            deriv = self.pend.dynamics(action)

            # Test the shape of the dynamics
            self.assertEqual(state.shape, deriv.shape)

            # Test that the value of the dynamics
            self.assertAllClose(deriv[:, 0], state[:, 1])
            self.assertAllClose(deriv[:, 1],
                                -tf.sin(state[:, 0]) + action[:, 0])

    def test_reset(self):
        ''' Test the reset method.
        '''
        with self.cached_session():
            self.pend.state = tf.random_normal(self.pend.state.shape)
            self.pend.reset()

            # Test that the state is reset properly.
            self.assertEqual(self.x0, self.pend.state)


if __name__ == '__main__':
    tf.test.main()
