''' Pendulum class.
'''
import tensorflow as tf
import numpy as np
from gym.spaces import Box
from .differentiable_env import DifferentiableEnv


class Pendulum(DifferentiableEnv):  #pylint: disable=too-many-instance-attributes
    '''
    A pendulum environment with a quadratic cost around the upright. The
    dynamics are integrated with forward Euler integration.
    '''

    def __init__(self, x0=None, dt=0.01, R=None, Q=None, Q_f=None):  #pylint: disable=too-many-arguments

        # Define the size of the inputs and outputs.
        self.num_actuators = 1
        self.num_states = 2

        # Initialize the initial state.
        self.x0 = x0 if x0 is not None else tf.constant([[0.0, 0.0]])
        assert self.x0.shape[-1] == self.num_states

        # Define cost terms.
        self.R = R if R is not None else dt * np.eye(
            self.num_actuators, dtype=np.float32)
        self.Q = Q if Q is not None else dt * np.eye(
            self.num_states, dtype=np.float32)
        self.Q_f = Q_f if Q_f is not None else np.eye(
            self.num_states, dtype=np.float32)
        self.goal = np.pi * tf.ones(2)

        # Define the action space.
        self.action_space = Box(
            np.array([-1]), np.array([1]), dtype=np.float32)

        super(Pendulum, self).__init__(x0=x0, dt=dt)

    def transition_cost(self, state, action):
        ''' The cost of being in a state and taking an action.
        '''
        with tf.name_scope('cost'):
            state_cost = tf.reduce_sum(
                tf.tensordot(state, self.Q, axes=[[-1], [0]]) * state)
            action_cost = tf.reduce_sum(
                tf.tensordot(action, self.R, axes=[[-1], [0]]) * action)

        return state_cost + action_cost

    def final_cost(self, state):
        ''' The cost of ending the simulation in a particular state.
        '''
        return tf.reduce_sum(
            tf.tensordot(state, self.Q_f, axes=[[-1], [0]]) * state)

    def reset(self):
        ''' Reset the pendulum to the zero state
        '''
        self.state = self.x0
        return self.state

    def dynamics(self, action):
        ''' Computes the state derivative.
        '''

        # Special case the non-vectorized version
        if len(self.state.shape) < 2:
            q = self.state[:1]
            dq = self.state[1:]
        else:
            q = self.state[:, :1]
            dq = self.state[:, 1:]

        d2q = -tf.sin(q) + action

        return tf.concat([dq, d2q], axis=-1)
