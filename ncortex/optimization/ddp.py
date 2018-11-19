''' Implementation of the Differential Dynamic Programming (DDP) Algorithm
'''

import autograd
import autograd.numpy as np


class DDP:  #pylint: disable=too-many-instance-attributes
    ''' A solver for the DDP algorithm.
    '''

    def __init__(self, env, x_0, u_init):
        ''' Initialize the DDP solver.
        '''

        self.n_time, self.n_u = u_init.shape
        self.n_x = x_0.shape

        # Generate step cost derivative functions.
        self.l_x = autograd.grad(env.transition_cost, 0)
        self.l_u = autograd.grad(env.transition_cost, 1)
        self.l_xx = autograd.grad(self.l_x, 0)
        self.l_xu = autograd.grad(self.l_x, 1)
        self.l_uu = autograd.grad(self.l_u, 1)

        # Generate dynamics derivative functions.
        self.f_x = autograd.grad(env.step, 0)
        self.f_u = autograd.grad(env.step, 1)
        self.f_xx = autograd.grad(self.f_x, 0)
        self.f_xu = autograd.grad(self.f_x, 1)
        self.f_uu = autograd.grad(self.f_u, 1)

    def forward(self):
        ''' The forward pass of the DDP algorithm.
        '''
        raise NotImplementedError

    def backward(self):
        ''' The backwards pass of the DDP algorithm.
        '''
        raise NotImplementedError

    def solve(self):
        ''' The backwards pass of the DDP algorithm.
        '''
        pass

def test_main():
    '''
    Test the DDP when running this file as a script.
    '''
    from ncortex.envs import Pendulum
    x_0 = np.zeros(2)
    u_init = np.zeros((100, 1))
    env = Pendulum(x_0, use_tf=False)
    ddp_solver = DDP(env, env.x_0, u_init)
    ddp_solver.solve()

if __name__ == "__main__":
    test_main()
