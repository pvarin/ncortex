''' Implementation of the Differential Dynamic Programming (DDP) Algorithm
'''

import autograd
import autograd.numpy as np
from ncortex.utils import is_pos_def


class DDP:  #pylint: disable=too-many-instance-attributes
    ''' A solver for the DDP algorithm.
    '''

    def __init__(self, env, x_init, u_init):
        ''' Initialize the DDP solver.
        '''

        self.env = env

        # Get number of states, controls and timesteps.
        self.n_steps, self.n_u = u_init.shape
        n_steps_x, self.n_x = x_init.shape
        assert self.n_steps + 1 == n_steps_x, \
            "Number of control steps must be one less than the number of states"
        
        # Allocate memory for relevant variables.
        self.x = x_init
        self.u = u_init
        self.du = np.empty_like(u_init)
        self.feedback = np.empty((self.n_steps, self.n_u, self.n_x))

        # Set initialization flags.
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

    def forward(self, stepsize=1):
        ''' The forward pass of the DDP algorithm.
        '''
        cost = 0
        u_proposed = np.empty_like(self.u)
        x_proposed = np.empty_like(self.x)
        x_proposed[0,:] = self.x[0,:]

        for i in range(self.n_steps):
            # Compute the action via the control law.
            x_err =  x_proposed[i, :] - self.x[i, :]
            u_proposed[i, :] = self.u[i, :] + stepsize * self.du[i, :] + self.feedback[
                i, :, :].dot(x_err)

            # Compute the transition cost.
            cost += self.env.transition_cost(x_proposed[i, :], u_proposed[i, :])

            # Evaluate the dynamics.
            x_proposed[i + 1, :] = self.env.step(x_proposed[i, :], u_proposed[i, :])

        # TODO: check the expected reward increase
        if True:
            # Accept the proposal
            self.u = u_proposed
            self.x = x_proposed
        else:
            return self.forward(.5*stepsize)

        # Add the final cost and return.
        cost += self.env.final_cost(x_proposed[-1, :])
        return cost

    def backward(self, reg=1e-2):  # pylint: disable=too-many-locals
        ''' The backwards pass of the DDP algorithm.
        '''

        # Start with the final cost
        v_x = self.l_final_x(self.x[-1, :])
        v_xx = self.l_final_xx(self.x[-1, :])

        for i in reversed(range(self.n_steps)):
            x = self.x[i, :]
            u = self.u[i, :]

            # Compute all of the relevant derivatives.
            l_x = self.l_x(x, u)
            l_u = self.l_u(x, u)
            l_xx = self.l_xx(x, u)
            l_xu = self.l_xu(x, u)
            l_uu = self.l_uu(x, u)

            f_x = self.f_x(x, u)
            f_u = self.f_u(x, u)
            f_xx = self.f_xx(x, u)
            f_xu = self.f_xu(x, u)
            f_uu = self.f_uu(x, u)

            # Compute the Q-function derivatives.
            q_x = l_x + np.einsum('i,ij->j', v_x, f_x)
            q_u = l_u + np.einsum('i,ij->j', v_x, f_u)
            q_xx = l_xx + np.einsum('ik,ij,kl->jl', v_xx, f_x, f_x) + \
                    np.einsum('i,ijk->jk', v_x, f_xx)
            q_xu = l_xu + np.einsum('ik,ij,kl->jl', v_xx, f_x, f_u) + \
                    np.einsum('i,ijk->jk', v_x, f_xu)
            reg = np.linalg.norm(q_u)
            q_uu = l_uu + np.einsum('ik,ij,kl->jl', v_xx, f_u, f_u) + \
                    np.einsum('i,ijk->jk', v_x, f_uu)# + \
            # reg*np.eye(self.n_u)
            # q_uu = (q_uu + q_uu.T)/2.

            if not is_pos_def(q_uu):
                print("Warning, q_uu is not positive definite.")
            # Regularize q_uu to make it positive definite.
            # auto_reg = np.maximum(reg, 1e-6)
            # print("regularizing q_uu")
            # while not is_pos_def(q_uu):
            # auto_reg *= 10
            # print("reg:  {}".format(auto_reg))
            # print("q_uu: {}".format(q_uu))
            # q_uu = q_uu + auto_reg*np.eye(self.n_u)
            # print("q_uu: {}".format(q_uu))

            # Solve for the feedforward and feedback terms using a single
            #   call to np.linalg.solve()
            res = np.linalg.solve(q_uu, np.hstack((q_u[:, np.newaxis],
                                                   q_xu.T)))
            self.du[i, :] = -res[:, 0]
            self.feedback[i, :, :] = -res[:, 1:]

            # Update the value function
            v_x = q_x + \
                np.einsum('ji,jk,k->i', self.feedback[i, :, :], q_uu, self.du[i, :]) + \
                np.einsum('ji,j->i', self.feedback[i, :, :], q_u) + \
                np.einsum('ij,j->i', q_xu, self.du[i,:])
            v_xx = q_xx + \
                np.einsum('ji,jk,kl->il', self.feedback[i, :, :], q_uu, self.feedback[i, :, :]) + \
                np.einsum('ji,kj->ik', self.feedback[i, :, :], q_xu) + \
                np.einsum('ij,jk->ik', q_xu, self.feedback[i, :, :])

        self.feedback_initialized = True

    def solve(self, max_iter=100, atol=1e-3, rtol=1e-3):
        ''' Solves the DDP algorithm to convergence.
        '''
        cost = []
        for i in range(max_iter):
            print(i)
            self.backward()
            next_cost = self.forward()
            cost.append(next_cost)

            # if (np.abs(cost[-1] - cost[-2]) < atol) and \
            #     (np.abs(cost[-1] - cost[-2])/cost[-1] < rtol):
            #     return cost

            last_cost = cost

        return cost
