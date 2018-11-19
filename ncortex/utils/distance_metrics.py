''' A module containing distance metrics for angles etc.
'''

from math import pi
import numpy as np
import tensorflow as tf


def angle_diff(theta1, theta2):
    ''' An angle subtraction method.
    '''
    if isinstance(theta1, tf.Tensor) or isinstance(theta2, tf.Tensor):
        delta = tf.mod(theta1 - theta2 - pi, 2 * pi) - pi
    else:
        delta = np.mod(theta1 - theta2 - pi, 2 * pi) - pi
    return delta


def angle_distance(theta1, theta2):
    ''' The equivalent of l-1 norm for angles.
    '''
    delta = angle_diff(theta1, theta2)
    if isinstance(delta, tf.Tensor):
        return tf.abs(delta)
    return np.abs(delta)


def squared_angle_distance(theta1, theta2):
    ''' The equivalent of l-2 norm for angles.
    '''
    delta = angle_diff(theta1, theta2)
    return delta**2
