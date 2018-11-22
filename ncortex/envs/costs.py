''' Helper classes for cost functions
'''
import tensorflow as tf
import autograd.numpy as np

def quadratic_cost(err, Q, use_tf=True):
    ''' Compute a quadratic cost.
    '''
    if use_tf:
        with tf.name_scope('cost'):
            if len(err.shape) == 1:
                return tf.einsum('i,ij,j', err, Q, err)

            # Special case the vectorized version.
            return tf.einsum('ki,ij,kj->k', err, Q, err)

    return np.einsum('...i,ij,...j', err, Q, err)
