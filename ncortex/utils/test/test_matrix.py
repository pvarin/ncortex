''' Test all of the functions in the utils module.
'''

import tensorflow as tf
import autograd.numpy as np
from ncortex.utils import (is_pos_def)


class TestMatrixUtils(tf.test.TestCase):
    ''' TestCase matrix utility functions.
    '''

    def test_is_pos_def(self):
        ''' Test is_pos_def function.
        '''
        mat = np.random.random((3, 3))
        mat = mat.T.dot(mat)
        self.assertTrue(is_pos_def(mat))
        self.assertFalse(is_pos_def(-mat))


if __name__ == '__main__':
    tf.test.main()
