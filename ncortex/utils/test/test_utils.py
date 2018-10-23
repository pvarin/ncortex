''' Test all of the functions in the utils module.
'''

import math
import tensorflow as tf
from ncortex.utils import (angle_distance, angle_diff, squared_angle_distance)


class TestAngleDistances(tf.test.TestCase):
    ''' TestCase for the angle distance functions using tensorflow's wrapper of unittest.
    '''

    def test_squared_angle_distance(self):
        ''' Test squared_angle_distance function.
        '''
        with self.cached_session():
            # Test that zero is exact.
            self.assertEqual(squared_angle_distance(0, 0).eval(), 0.0)

            # Test an arbitrary value.
            self.assertAllClose(squared_angle_distance(-1.0, 2.0).eval(), 9.0)

            # Test that is wraps with additive multiples of 2 pi.
            self.assertAllClose(
                squared_angle_distance(-1.0, 2.0 + 2 * math.pi).eval(), 9.0)
            self.assertAllClose(
                squared_angle_distance(-1.0, 2.0 + 4 * math.pi).eval(), 9.0)

            # Test the gradient.
            theta_1 = tf.constant(-1.0)
            theta_2 = tf.constant(2.0)
            grad = tf.gradients(
                squared_angle_distance(theta_1, theta_2), [theta_1, theta_2])
            self.assertAllClose(grad, [-6.0, 6.0])

    def test_angle_diff(self):
        ''' Test angle_diff function.
        '''
        with self.cached_session():
            # Test that zero is exact.
            self.assertEqual(angle_diff(0, 0).eval(), 0.0)

            # Test an arbitrary value and the reverse.
            self.assertAllClose(angle_diff(-1.0, 2.0).eval(), -3.0)
            self.assertAllClose(angle_diff(2.0, -1.0).eval(), 3.0)

            # Test that is wraps with additive multiples of 2 pi.
            self.assertAllClose(
                angle_diff(-1.0, 2.0 + 2 * math.pi).eval(), -3.0)
            self.assertAllClose(
                angle_diff(-1.0, 2.0 + 4 * math.pi).eval(), -3.0)

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
            self.assertEqual(angle_distance(0, 0).eval(), 0.0)

            # Test an arbitrary value and the reverse.
            self.assertAllClose(angle_distance(-1.0, 2.0).eval(), 3.0)
            self.assertAllClose(angle_distance(2.0, -1.0).eval(), 3.0)

            # Test that is wraps with additive multiples of 2 pi.
            self.assertAllClose(
                angle_distance(-1.0, 2.0 + 2 * math.pi).eval(), 3.0)
            self.assertAllClose(
                angle_distance(-1.0, 2.0 + 4 * math.pi).eval(), 3.0)

            # Test the gradient.
            theta_1 = tf.constant(-1.0)
            theta_2 = tf.constant(2.0)
            grad = tf.gradients(
                angle_distance(theta_1, theta_2), [theta_1, theta_2])
            self.assertAllClose(grad, [-1.0, 1.0])


if __name__ == '__main__':
    tf.test.main()
