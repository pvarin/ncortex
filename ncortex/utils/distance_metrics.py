''' A module containing distance metrics for angles etc.
'''

from math import pi
import tensorflow as tf


def angle_diff(theta1, theta2):
    ''' An angle subtraction method.
    '''
    delta = tf.mod(theta1 - theta2 - pi, 2 * pi) - pi
    return delta


def angle_distance(theta1, theta2):
    ''' The equivalent of l-1 norm for angles.
    '''
    delta = angle_diff(theta1, theta2)
    return tf.abs(delta)


def squared_angle_distance(theta1, theta2):
    ''' The equivalent of l-2 norm for angles.
    '''
    delta = angle_diff(theta1, theta2)
    return delta**2
