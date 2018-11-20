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
        self.x[0, :] = x_0
        self.u = u_init
        self.feedback_gains = np.empty((self.n_steps, self.n_u, self.n_x))

        # Set initialization flags.
        self.state_initialized = False
        self.feedback_initialized = False

        # Generate one-step cost derivative functions.
        self.l_x = autograd.grad(env.transition_cost, 0)
        self.l_u = autograd.grad(env.transition_cost, 1)
        self.l_xx = autograd.jacobian(self.l_x, 0)
        self.l_xu = autograd.jacobian(self.l_x, 1)
        self.l_uu = autograd.jacobian(self.l_u, 1)

        # Generate the final cost derivative functions.
        self.l_final_x = autograd.grad(env.final_cost, 0)
        self.l_final_xx = autograd.jacobian(self.l_final_x, 0)

        # Generate dynamics derivative functions.
        self.f_x = autograd.jacobian(env.step, 0)
        self.f_u = autograd.jacobian(env.step, 1)
        self.f_xx = autograd.jacobian(self.f_x, 0)
        self.f_xu = autograd.jacobian(self.f_x, 1)
        self.f_uu = autograd.jacobian(self.f_u, 1)

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

    def backward(self): # pylint: disable=too-many-locals
        ''' The backwards pass of the DDP algorithm.
        '''
        assert self.state_initialized, \
            "The forward pass must be run before the backwards pass"

        # Start with the final cost
        v_x = self.l_final_x(self.x[-1, :])
        v_xx = self.l_final_xx(self.x[-1, :])

        for i in reversed(range(self.n_steps)):
            # Compute all of the relevant derivatives.
            l_x = self.l_x(self.x[i, :], self.u[i, :])
            l_u = self.l_u(self.x[i, :], self.u[i, :])
            l_xx = self.l_xx(self.x[i, :], self.u[i, :])
            l_xu = self.l_xu(self.x[i, :], self.u[i, :])
            l_uu = self.l_uu(self.x[i, :], self.u[i, :])

            f_x = self.f_x(self.x[i, :], self.u[i, :])
            f_u = self.f_u(self.x[i, :], self.u[i, :])
            f_xx = self.f_xx(self.x[i, :], self.u[i, :])
            f_xu = self.f_xu(self.x[i, :], self.u[i, :])
            f_uu = self.f_uu(self.x[i, :], self.u[i, :])

            # Compute the Q-function derivatives.
            q_x = l_x + np.einsum('i,ij->j', v_x, f_x)
            q_u = l_u + np.einsum('i,ij->j', v_x, f_u)
            q_xx = l_xx + np.einsum('i,ijk->jk', v_x, f_xx) + \
                    np.einsum('ik,ij,kl->jl', v_xx, f_x, f_x)
            q_xu = l_xu + np.einsum('i,ijk->jk', v_x, f_xu) + \
                    np.einsum('ik,ij,kl->jl', v_xx, f_x, f_u)
            q_uu = l_uu + np.einsum('i,ijk->jk', v_x, f_uu) + \
                    np.einsum('ik,ij,kl->jl', v_xx, f_u, f_u)

            # Solve for the feedforward and feedback terms using a single
            #   call to np.linalg.solve()
            res = np.linalg.solve(q_uu, np.hstack((q_u[:, np.newaxis], q_xu.T)))
            self.u[i, :] = res[:, 0]
            self.feedback_gains[i, :, :] = res[:, 1:]

            # Update the value function
            v_x = q_x - np.einsum('ji,ik,k->j', q_xu, q_uu, q_u)
            v_xx = q_xx - np.einsum('ji,ik,lk->jl', q_xu, q_uu, q_xu)

        self.feedback_initialized = True

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
