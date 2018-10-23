''' Pendulum class.
'''
import tensorflow as tf
import numpy as np


class Pendulum:
    '''
    A pendulum environment with a quadratic cost around the upright. The
    dynamics are integrated with forward Euler integration.
    '''

    def __init__(self, R=None, Q=None, Q_f=None, dt=0.01):
        self.num_actuators = 1
        self.num_states = 2
        self.dt = dt

        self.R = R if R is not None else dt * np.eye(self.num_actuators)
        self.Q = Q if Q is not None else dt * np.eye(self.num_states)
        self.Q_f = Q_f if Q_f is not None else np.eye(self.num_states)
        self.goal = np.pi * tf.ones(2)

    def transition_cost(self, state, action):
        ''' The cost of being in a state and taking an action.
        '''
        with tf.name_scope('cost'):
            return tf.matmul(tf.matmul(state, self.Q), state) + \
            tf.matmul(tf.matmul(action, self.R), action)

    def final_cost(self, state):
        ''' The cost of ending the simulation in a state.
        '''
        return state.dot(self.Q_f).dot(self.Q_f)

    def step(self, state, action):
        ''' Takes an action in a particular state and returns the next state
        '''
        d2q = -tf.sin(state[0]) + action
        return state + [self.dt * state[1], self.dt * d2q]
