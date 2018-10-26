''' An example that runs and renders a pendulum agent.
'''
import time
import numpy as np
from ncortex.envs import Pendulum

if __name__ == '__main__':
    x0 = np.array([1., 0.])
    NO_TORQUE = np.array([0.])
    env = Pendulum(x0=x0)
    for i in range(1000):
        env.step(NO_TORQUE)
        env.render()
        time.sleep(env.dt)
