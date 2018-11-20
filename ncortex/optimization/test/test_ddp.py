''' Test the trajectory optimization procedures.
'''

import tensorflow as tf
import autograd.numpy as np
from ncortex.envs import Pendulum
from ncortex.optimization.ddp import DDP


class TestDDP(tf.test.TestCase):
    ''' Test the trajectory optimization procedures.
    '''

    def setUp(self):
        ''' Setup each test case with DDP for the pendulum.
        '''
        env = Pendulum(use_tf=False)
        x_0 = np.zeros(2)
        u_init = np.zeros((100, 1))
        self.ddp = DDP(env, x_0, u_init)

    def test_final_cost_derivatives(self):
        ''' Test the final cost derivatives with finite differences.
        '''
        state = np.array([.1, .2])
        l_final = self.ddp.env.final_cost(state)
        l_final_x = self.ddp.l_final_x(state)
        l_final_xx = self.ddp.l_final_xx(state)

        # Test the first derivative.
        l_final_x_approx = np.empty(2)
        eps = 1e-6
        for i in range(2):
            d_state = np.zeros(2)
            d_state[i] += eps
            l_final_x_approx[i] = (
                self.ddp.env.final_cost(state + d_state) - l_final) / eps

        self.assertAllClose(l_final_x, l_final_x_approx)

        # Test the second derivative.
        l_final_xx_approx = np.empty((2, 2))
        eps = 1e-6
        for i in range(2):
            d_state = np.zeros(2)
            d_state[i] += eps
            l_final_xx_approx[:, i] = (
                self.ddp.l_final_x(state + d_state) - l_final_x) / eps

        self.assertAllClose(l_final_xx, l_final_xx_approx)

    def test_transition_cost_derivatives(self): # pylint: disable=too-many-locals
        ''' Test the transition cost derivatives with finite differences.
        '''
        state = np.array([.1, .2])
        action = np.array([.3])
        l_true = self.ddp.env.transition_cost(state, action)
        l_x = self.ddp.l_x(state, action)
        l_u = self.ddp.l_u(state, action)
        l_xx = self.ddp.l_xx(state, action)
        l_xu = self.ddp.l_xu(state, action)
        l_uu = self.ddp.l_uu(state, action)

        # Test the first derivatives.
        l_x_approx = np.empty(2)
        eps = 1e-6
        for i in range(2):
            d_state = np.zeros(2)
            d_state[i] += eps
            l_x_approx[i] = (self.ddp.env.transition_cost(
                state + d_state, action) - l_true) / eps

        self.assertAllClose(l_x, l_x_approx)

        l_u_approx = np.empty(1)
        for i in range(1):
            d_action = np.zeros(1)
            d_action[i] += eps
            l_u_approx[i] = (self.ddp.env.transition_cost(
                state, action + d_action) - l_true) / eps

        self.assertAllClose(l_u, l_u_approx)

        # Test the second derivatives.
        l_xx_approx = np.empty((2, 2))
        eps = 1e-6
        for i in range(2):
            d_state = np.zeros(2)
            d_state[i] += eps
            l_xx_approx[:, i] = (
                self.ddp.l_x(state + d_state, action) - l_x) / eps

        self.assertAllClose(l_xx, l_xx_approx)

        l_xu_approx = np.empty((2, 1))
        for i in range(1):
            d_action = np.zeros(1)
            d_action[i] += eps
            l_xu_approx[:, i] = (
                self.ddp.l_x(state, action + d_action) - l_x) / eps

        self.assertAllClose(l_xu, l_xu_approx)

        l_uu_approx = np.empty((1, 1))
        for i in range(1):
            d_action = np.zeros(1)
            d_action[i] += eps
            l_uu_approx[:, i] = (
                self.ddp.l_u(state, action + d_action) - l_u) / eps

        self.assertAllClose(l_uu, l_uu_approx)

    def test_dynamics_derivatives(self): # pylint: disable=too-many-locals
        ''' Test the transition cost derivatives with finite differences.
        '''
        state = np.array([.1, .2])
        action = np.array([.3])
        x_next = self.ddp.env.step(state, action)
        f_x = self.ddp.f_x(state, action)
        f_u = self.ddp.f_u(state, action)
        f_xx = self.ddp.f_xx(state, action)
        f_xu = self.ddp.f_xu(state, action)
        f_uu = self.ddp.f_uu(state, action)

        self.assertEqual(f_x.shape, (2, 2))
        self.assertEqual(f_u.shape, (2, 1))
        self.assertEqual(f_xx.shape, (2, 2, 2))
        self.assertEqual(f_xu.shape, (2, 2, 1))
        self.assertEqual(f_uu.shape, (2, 1, 1))

        # Test the first derivatives.
        f_x_approx = np.empty((2, 2))
        eps = 1e-6
        for i in range(2):
            d_state = np.zeros(2)
            d_state[i] += eps
            f_x_approx[:, i] = (
                self.ddp.env.step(state + d_state, action) - x_next) / eps

        self.assertAllClose(f_x, f_x_approx)

        f_u_approx = np.empty((2, 1))
        for i in range(1):
            d_action = np.zeros(1)
            d_action[i] += eps
            f_u_approx[:, i] = (
                self.ddp.env.step(state, action + d_action) - x_next) / eps

        self.assertAllClose(f_u, f_u_approx)

        # Test the second derivatives.
        f_xx_approx = np.empty((2, 2, 2))
        eps = 1e-6
        for i in range(2):
            d_state = np.zeros(2)
            d_state[i] += eps
            f_xx_approx[:, :, i] = (
                self.ddp.f_x(state + d_state, action) - f_x) / eps

        self.assertAllClose(f_xx, f_xx_approx)

        f_xu_approx = np.empty((2, 2, 1))
        for i in range(1):
            d_action = np.zeros(1)
            d_action[i] += eps
            f_xu_approx[:, :, i] = (
                self.ddp.f_x(state, action + d_action) - f_x) / eps

        self.assertAllClose(f_xu, f_xu_approx)

        f_uu_approx = np.empty((2, 1, 1))
        for i in range(1):
            d_action = np.zeros(1)
            d_action[i] += eps
            f_uu_approx[:, :, i] = (
                self.ddp.f_u(state, action + d_action) - f_u) / eps

        self.assertAllClose(f_uu, f_uu_approx)

    def test_forward(self):
        ''' Test the DDP forward pass.
        '''
        self.ddp.forward()

    def test_backward(self):
        ''' Test the DDP backward pass.
        '''
        with self.assertRaises(Exception):
            self.ddp.backward()

        self.ddp.forward()
        self.ddp.backward()


if __name__ == '__main__':
    tf.test.main()
