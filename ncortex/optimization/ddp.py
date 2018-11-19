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

        self.env = env

        # Allocate memory for relevant variables.
        self.n_steps, self.n_u = u_init.shape
        self.n_x, = x_0.shape
        self.x = np.empty((self.n_steps + 1, self.n_x))
        self.x[:, 0] = x_0
        self.u = u_init
        self.feedback_gains = np.empty((self.n_steps, self.n_u, self.n_x))

        # Set initialization flags.
        self.state_initialized = False
        self.feedback_initialized = False

        # Generate one-step cost derivative functions.
        self.l_x = autograd.grad(env.transition_cost, 0)
        self.l_u = autograd.grad(env.transition_cost, 1)
        self.l_xx = autograd.grad(self.l_x, 0)
        self.l_xu = autograd.grad(self.l_x, 1)
        self.l_uu = autograd.grad(self.l_u, 1)

        # Generate the final cost derivative functions.
        self.l_final_x = autograd.grad(env.final_cost, 0)
        self.l_final_xx = autograd.grad(self.l_final_x, 0)

        # Generate dynamics derivative functions.
        self.f_x = autograd.grad(env.step, 0)
        self.f_u = autograd.grad(env.step, 1)
        self.f_xx = autograd.grad(self.f_x, 0)
        self.f_xu = autograd.grad(self.f_x, 1)
        self.f_uu = autograd.grad(self.f_u, 1)

    def forward(self):
        ''' The forward pass of the DDP algorithm.
        '''
        if not self.feedback_initialized:
            for i in range(self.n_steps):
                # Evaluate the dynamics with a bling feedforward torque
                self.x[i + 1, :] = self.env.step(self.x[i, :], self.u[i, :])
        else:
            x_target = self.x
            for i in range(self.n_steps):
                # Compute the action via the control law.
                x_err = x_target[i, :] - self.x[i, :]
                ctrl = self.u[i, :] + self.feedback_gains[i, :, :].dot(x_err)

                # Evaluate the dynamics.
                self.x[i + 1, :] = self.env.step(self.x[i, :], ctrl)

        self.state_initialized = True

    def backward(self):
        ''' The backwards pass of the DDP algorithm.
        '''
        assert self.state_initialized, \
            "The forward pass must be run before the backwards pass"

        self.feedback_initialized = True
        for _ in reversed(range(self.n_steps)):
            pass
            # v_x = self.l_final_x(self.x[i])
            # v_xx = self.l_final_xx(self.x[i])

            # TODO: compute the Q terms
            # TODO: compute the feedforward/feedback terms from the Q terms

        raise NotImplementedError

    def solve(self):
        ''' Solves the DDP algorithm to convergence.
        '''
        pass


def test_main():
    '''
    Test DDP when running this file as a script.
    '''
    from ncortex.envs import Pendulum
    x_0 = np.zeros(2)
    u_init = np.zeros((100, 1))
    env = Pendulum(x_0, use_tf=False)
    ddp_solver = DDP(env, env.x_0, u_init)
    ddp_solver.solve()


if __name__ == "__main__":
    test_main()
