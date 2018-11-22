''' Some matrix utilities.
'''

import autograd.numpy as np


def is_pos_def(mat):
    ''' Checks if a matrix is symmetric positive definite.
    '''
    if np.allclose(mat, mat.T):
        try:
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return is_pos_def(mat + mat.T)
