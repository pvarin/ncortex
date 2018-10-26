''' An example that runs and renders a pendulum agent.
'''
import time
import numpy as np
from ncortex.envs import Pendulum

if __name__ == '__main__':
    # Initialize the Pendulum environment
    x0 = np.array([1., 0.])
    env = Pendulum(x0=x0)

    # Simulate for 300 timesteps
    for i in range(300):
        env.step(np.array([0.]))
        env.render()
        time.sleep(env.dt)
