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

    def forward(self, last_cost=None, dv1=None, dv2=None, stepsize=1.):
        ''' The forward pass of the DDP algorithm.
        '''
        cost = 0
        u_proposed = np.empty_like(self.u)
        x_proposed = np.empty_like(self.x)
        x_proposed[0, :] = self.x[0, :]

        for i in range(self.n_steps):
            # Compute the action via the control law.
            x_err = x_proposed[i, :] - self.x[i, :]
            u_proposed[i, :] = self.u[i, :] + stepsize * self.du[
                i, :] + self.feedback[i, :, :].dot(x_err)

            # Compute the transition cost.
            cost += self.env.transition_cost(x_proposed[i, :],
                                             u_proposed[i, :])

            # Evaluate the dynamics.
            x_proposed[i + 1, :] = self.env.step(x_proposed[i, :],
                                                 u_proposed[i, :])

        # Add the final cost and return.
        cost += self.env.final_cost(x_proposed[-1, :])


        # Accept if there is no prior cost.
        if last_cost is None:
            self.u = u_proposed
            self.x = x_proposed
            return cost, stepsize

        # Check the linesearch termination condition.
        relative_improvement = (cost - last_cost)/(stepsize * dv1 + stepsize**2 * dv2)
        if relative_improvement > .1:
            # Accept the proposal.
            self.u = u_proposed
            self.x = x_proposed
            return cost, stepsize

        # Reduce the stepsize and recurse.
        return self.forward(
            last_cost=last_cost, dv1=dv1, dv2=dv2, stepsize=.5 * stepsize)

    def backward(self, reg=1e-1):  # pylint: disable=too-many-locals
        ''' The backwards pass of the DDP algorithm.
        '''

        # Start with the final cost
        dv1 = 0
        dv2 = 0
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
            q_uu = l_uu + np.einsum('ik,ij,kl->jl', v_xx, f_u, f_u) + \
                    np.einsum('i,ijk->jk', v_x, f_uu)

            # Compute the regularized Q-function.
            q_xu_reg = q_xu + reg * np.einsum('ji,jk->ik', f_x, f_u)
            q_uu_reg = q_uu + reg * np.einsum('ji,jk->ik', f_u, f_u)

            # Regularize q_uu to make it positive definite.
            if not is_pos_def(q_uu_reg):
                print("Step {}:\nReg: {}".format(i, reg))
                print(
                    "Not Quu is not PSD, regularizing and restarting backwards pass"
                )
                return self.backward(reg=2. * reg)

            # Solve for the feedforward and feedback terms using a single
            #   call to np.linalg.solve()
            res = np.linalg.solve(q_uu_reg,
                                  np.hstack((q_u[:, np.newaxis], q_xu_reg.T)))
            self.du[i, :] = -res[:, 0]
            self.feedback[i, :, :] = -res[:, 1:]

            # Update the value function
            dv1 += np.einsum('i,i', self.du[i, :], q_u)
            dv2 += .5 * np.einsum('i,ij,j', self.du[i, :], q_uu, self.du[i, :])
            v_x = q_x + \
                np.einsum('ji,jk,k->i', self.feedback[i, :, :], q_uu, self.du[i, :]) + \
                np.einsum('ji,j->i', self.feedback[i, :, :], q_u) + \
                np.einsum('ij,j->i', q_xu, self.du[i, :])
            v_xx = q_xx + \
                np.einsum('ji,jk,kl->il', self.feedback[i, :, :], q_uu, self.feedback[i, :, :]) + \
                np.einsum('ji,kj->ik', self.feedback[i, :, :], q_xu) + \
                np.einsum('ij,jk->ik', q_xu, self.feedback[i, :, :])

        return dv1, dv2, reg

    def solve(self, max_iter=100, atol=1e-6, rtol=1e-6):
        ''' Solves the DDP algorithm to convergence.
        '''
        info = {
            'cost': [],
            'stepsize': [],
            'reg': [],
            'norm_du': [],
            'norm_du_relative': []
        }
        last_cost = None
        reg = 1e-1
        for i in range(max_iter):
            print(i)
            # Backward pass with adaptive regularizer.
            reg /= 2.
            dv1, dv2, reg = self.backward(reg=reg)

            # Forward pass with linesearch.
            last_cost, stepsize = self.forward(
                last_cost=last_cost, dv1=dv1, dv2=dv2)

            # Log relevant info.
            info['cost'].append(last_cost)
            info['reg'].append(reg)
            info['stepsize'].append(stepsize)
            info['norm_du'].append(np.linalg.norm(self.du))
            info['norm_du_relative'].append(
                np.linalg.norm(self.du) / np.linalg.norm(self.u))

            if info['norm_du'][-1] < atol or info['norm_du_relative'][-1] < rtol:
                # Terminate if the change in control is sufficiently small.
                return info

        return info
