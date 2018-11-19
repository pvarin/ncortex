''' An example that runs and renders a pendulum agent.
'''
import time
import numpy as np
from ncortex.envs import Pendulum

if __name__ == '__main__':
    # Initialize the Pendulum environment
    x = np.array([1., 0.])
    env = Pendulum(x_0=x, use_tf=False)

    # Simulate for 300 timesteps
    for i in range(300):
        x = env.step(x, np.array([0.]))
        env.render(x)
        time.sleep(env.dt)
