''' Test all of the functions in the utils module.
'''

import math
import tensorflow as tf
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

    def test_angle_diff(self):
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

    def test_angle_distance(self):
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


if __name__ == '__main__':
    tf.test.main()
