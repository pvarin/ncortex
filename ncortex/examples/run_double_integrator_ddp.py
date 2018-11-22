''' An example that runs and renders a pendulum agent.
'''
import autograd.numpy as np
from ncortex.envs import DoubleIntegrator
from ncortex.optimization.ddp import DDP
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Initialize the Pendulum environment
    x_init = np.random.random((101, 2))
    # x_0 = np.array([1., 0.])
    u_init = np.zeros((100, 1))
    env = DoubleIntegrator(dt=0.05, x_0=x_init[0,:], use_tf=False)
    ddp = DDP(env, x_init, u_init)
    cost = ddp.solve(max_iter=10)
    plt.plot(ddp.x[:,0], ddp.x[:,1])
    plt.figure()
    plt.plot(cost)
    plt.show()