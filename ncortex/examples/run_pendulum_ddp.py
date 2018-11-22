''' An example that runs and renders a pendulum agent.
'''
import autograd.numpy as np
import matplotlib.pyplot as plt
from ncortex.envs import Pendulum
from ncortex.optimization.ddp import DDP


def main():
    ''' Run an example DDP on the Pendulum environment.
    '''
    # Initialize the environment and solve with DDP.
    x_init = np.random.random((101, 1))
    u_init = np.zeros((100, 1))
    env = Pendulum(dt=0.05, x_0=x_init[0, :], use_tf=False)
    ddp = DDP(env, x_init, u_init)
    cost = ddp.solve(max_iter=10)

    # Plot the solution in phase space.
    plt.plot(ddp.x[:, 0], ddp.x[:, 1])
    plt.figure()
    plt.plot(cost)
    plt.show()


if __name__ == '__main__':
    main()
