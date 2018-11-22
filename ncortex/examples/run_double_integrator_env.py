''' An example that runs and renders a pendulum agent.
'''
import time
import autograd.numpy as np
from ncortex.envs import DoubleIntegrator

if __name__ == '__main__':
    # Initialize the Pendulum environment
    x = np.array([0., 0.])
    env = DoubleIntegrator(x_0=x, use_tf=False)

    # Simulate for 300 timesteps
    N = 3000
    t = env.dt*np.arange(N)
    for i in range(N):
        x = env.step(x, np.cos(t[i,np.newaxis]))
        env.render(x)
        time.sleep(env.dt)
