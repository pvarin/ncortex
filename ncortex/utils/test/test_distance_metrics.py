''' Test all of the functions in the utils module.
'''

import math
import tensorflow as tf
import autograd
import autograd.numpy as np
from ncortex.utils import (angle_distance, angle_diff, squared_angle_distance)


class TestAngleDistances(tf.test.TestCase):
    ''' TestCase for the angle distance functions using tensorflow's wrapper of unittest.
    '''

    def test_squared_angle_distance_tf(self):
        ''' Test squared_angle_distance function.
        '''
        with self.cached_session():
            # Test that zero is exact.
            theta_1 = tf.constant(0.)
            theta_2 = tf.constant(0.)
            self.assertEqual(squared_angle_distance(theta_1, theta_2).eval(), 0.0)

            # Test an arbitrary value.
            theta_1 = tf.constant(-1.)
            theta_2 = tf.constant(2.)
            self.assertAllClose(squared_angle_distance(theta_1, theta_2).eval(), 9.0)

            # Test that is wraps with additive multiples of 2 pi.
            theta_2 = tf.constant(2. + 2 * math.pi)
            self.assertAllClose(
                squared_angle_distance(theta_1, theta_2).eval(), 9.0)
            theta_2 = tf.constant(2. + 4 * math.pi)
            self.assertAllClose(
                squared_angle_distance(theta_1, theta_2).eval(), 9.0)

            # Test the gradient.
            theta_1 = tf.constant(-1.)
            theta_2 = tf.constant(2.)
            grad = tf.gradients(
                squared_angle_distance(theta_1, theta_2), [theta_1, theta_2])
            self.assertAllClose(grad, [-6., 6.])

    def test_squared_angle_distance_np(self):
        ''' Test squared_angle_distance function.
        '''
        # Test that zero is exact.
        self.assertEqual(squared_angle_distance(0., 0.), 0.0)

        # Test an arbitrary value.
        self.assertAllClose(squared_angle_distance(-1., 2.), 9.0)

        # Test that is wraps with additive multiples of 2 pi.
        self.assertAllClose(
            squared_angle_distance(-1., 2. + 2 * math.pi), 9.0)
        theta_2 = tf.constant(2. + 4 * math.pi)
        self.assertAllClose(
            squared_angle_distance(-1., 2. + 4 * math.pi), 9.0)

        # Test the gradient.
        theta_1 = np.array(-1.)
        theta_2 = np.array(2.)
        grad_arg0_squared_angle_distance = autograd.grad(squared_angle_distance, 0)
        grad_arg1_squared_angle_distance = autograd.grad(squared_angle_distance, 1)
        self.assertAllClose(grad_arg0_squared_angle_distance(theta_1, theta_2), -6.)
        self.assertAllClose(grad_arg1_squared_angle_distance(theta_1, theta_2), 6.)

    def test_angle_diff_tf(self):
        ''' Test angle_diff function.
        '''
        with self.cached_session():
            # Test that zero is exact.
            theta_1 = tf.constant(0.)
            theta_2 = tf.constant(0.)
            self.assertEqual(angle_diff(theta_1, theta_2).eval(), 0.0)

            # Test an arbitrary value and the reverse.
            theta_1 = tf.constant(-1.)
            theta_2 = tf.constant(2.)
            self.assertAllClose(angle_diff(theta_1, theta_2).eval(), -3.0)
            self.assertAllClose(angle_diff(theta_2, theta_1).eval(), 3.0)

            # Test that is wraps with additive multiples of 2 pi.
            theta_2 = tf.constant(2. + 2 * math.pi)
            self.assertAllClose(
                angle_diff(theta_1, theta_2).eval(), -3.0)
            theta_2 = tf.constant(2. + 4 * math.pi)
            self.assertAllClose(
                angle_diff(theta_1, theta_2).eval(), -3.0)

            # Test the gradient.
            theta_1 = tf.constant(-1.0)
            theta_2 = tf.constant(2.0)
            grad = tf.gradients(
                angle_diff(theta_1, theta_2), [theta_1, theta_2])
            self.assertAllClose(grad, [1.0, -1.0])

    def test_angle_diff_np(self):
        ''' Test angle_diff function.
        '''
        # Test that zero is exact.
        self.assertEqual(angle_diff(0., 0.), 0.0)

        # Test an arbitrary value and the reverse.
        self.assertAllClose(angle_diff(-1., 2.), -3.0)
        self.assertAllClose(angle_diff(2., -1.), 3.0)

        # Test that is wraps with additive multiples of 2 pi.
        self.assertAllClose(
            angle_diff(-1., 2. + 2 * math.pi), -3.0)
        self.assertAllClose(
            angle_diff(-1., 2. + 4 * math.pi), -3.0)

        # Test the gradient.
        theta_1 = np.array(-1.0)
        theta_2 = np.array(2.0)
        grad_arg0_angle_diff = autograd.grad(angle_diff, 0)
        grad_arg1_angle_diff = autograd.grad(angle_diff, 1)
        self.assertAllClose(grad_arg0_angle_diff(theta_1, theta_2), 1.0)
        self.assertAllClose(grad_arg1_angle_diff(theta_1, theta_2), -1.0)

    def test_angle_distance_tf(self):
        ''' Test angle_distance function.
        '''
        with self.cached_session():
            # Test that zero is exact.
            theta_1 = tf.constant(0.)
            theta_2 = tf.constant(0.)
            self.assertEqual(angle_distance(theta_1, theta_2).eval(), 0.0)

            # Test an arbitrary value and the reverse.
            theta_1 = tf.constant(-1.)
            theta_2 = tf.constant(2.)
            self.assertAllClose(angle_distance(theta_1, theta_2).eval(), 3.0)
            self.assertAllClose(angle_distance(theta_2, theta_1).eval(), 3.0)

            # Test that is wraps with additive multiples of 2 pi.
            theta_2 = tf.constant(2.0 + 2 * math.pi)
            self.assertAllClose(
                angle_distance(theta_1, theta_2).eval(), 3.0)
            theta_2 = tf.constant(2.0 + 4 * math.pi)
            self.assertAllClose(
                angle_distance(theta_1, theta_2).eval(), 3.0)

            # Test the gradient.
            theta_1 = tf.constant(-1.0)
            theta_2 = tf.constant(2.0)
            grad = tf.gradients(
                angle_distance(theta_1, theta_2), [theta_1, theta_2])
            self.assertAllClose(grad, [-1.0, 1.0])

    def test_angle_distance_np(self):
        ''' Test angle_distance function.
        '''
        # Test that zero is exact.
        self.assertEqual(angle_distance(0., 0.), 0.0)

        # Test an arbitrary value and the reverse.
        self.assertAllClose(angle_distance(-1., 2.), 3.0)
        self.assertAllClose(angle_distance(2., -1.), 3.0)

        # Test that is wraps with additive multiples of 2 pi.
        self.assertAllClose(
            angle_distance(-1., 2.0 + 2 * math.pi), 3.0)
        self.assertAllClose(
            angle_distance(-1., 2.0 + 4 * math.pi), 3.0)

        # # Test the gradient.
        theta_1 = np.array(-1.0)
        theta_2 = np.array(2.0)
        grad_arg0_angle_distance = autograd.grad(angle_distance, 0)
        grad_arg1_angle_distance = autograd.grad(angle_distance, 1)
        self.assertAllClose(grad_arg0_angle_distance(theta_1, theta_2), -1.0)
        self.assertAllClose(grad_arg1_angle_distance(theta_1, theta_2), 1.0)


if __name__ == '__main__':
    tf.test.main()
