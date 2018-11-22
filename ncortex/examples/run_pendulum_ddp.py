''' An example that runs and renders a pendulum agent.
'''
import autograd.numpy as np
from ncortex.envs import Pendulum
from ncortex.optimization.ddp import DDP
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Initialize the Pendulum environment
    x_0 = np.zeros(2)
    # x_0 = np.array([1., 0.])
    u_init = np.zeros((100, 1))
    env = Pendulum(dt=0.05, x_0=x_0, use_tf=False)
    ddp = DDP(env, x_0, u_init)
    cost = ddp.solve(max_iter=10)
    plt.plot(ddp.x[:,0], ddp.x[:,1])
    plt.figure()
    plt.plot(cost)
    plt.show()