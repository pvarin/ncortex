''' Pendulum class.
'''
import tensorflow as tf
import numpy as np
from gym.spaces import Box
from .differentiable_env import DifferentiableEnv


class Pendulum(DifferentiableEnv): #pylint: disable=too-many-instance-attributes
    '''
    A pendulum environment with a quadratic cost around the upright. The
    dynamics are integrated with forward Euler integration.
    '''

    def __init__(self, x0=None, dt=0.01, R=None, Q=None, Q_f=None): #pylint: disable=too-many-arguments

        # Define the size of the inputs and outputs.
        self.num_actuators = 1
        self.num_states = 2

        # Initialize the initial state.
        self.x0 = x0 if x0 is not None else tf.constant([[0.0, 0.0]])
        assert self.x0.shape[-1] == self.num_states

        # Define cost terms.
        self.R = R if R is not None else dt * np.eye(self.num_actuators, dtype=np.float32)
        self.Q = Q if Q is not None else dt * np.eye(self.num_states, dtype=np.float32)
        self.Q_f = Q_f if Q_f is not None else np.eye(self.num_states, dtype=np.float32)
        self.goal = np.pi * tf.ones(2)

        # Define the action space.
        self._action_space = Box(
            np.array([-1]), np.array([1]), dtype=np.float32)

        super(Pendulum, self).__init__(x0=x0, dt=dt)

    @property
    def action_space(self):
        ''' Implement the action_space property.
        '''
        return self._action_space

    def transition_cost(self, state, action):
        ''' The cost of being in a state and taking an action.
        '''
        with tf.name_scope('cost'):
            return tf.matmul(tf.matmul(state, self.Q, transpose_a=True), state) + \
                   tf.matmul(tf.matmul(action, self.R, transpose_a=True), action)

    def final_cost(self, state):
        ''' The cost of ending the simulation in a particular state.
        '''
        return state.dot(self.Q_f).dot(self.Q_f)

    def set_state(self, state):
        ''' Set the state of the environment manually.
        '''
        self._state = state

    def reset(self):
        ''' Reset the pendulum to the zero state
        '''
        self._state = self.x0
        return self._state

    def dynamics(self, action):
        ''' Computes the state derivative.
        '''
        d2q = -tf.sin(self._state[:, 0]) + action[:, 0]
        return tf.concat([self._state[:, 1], d2q], axis=-1)
