''' Pendulum class.
'''
import tensorflow as tf
import numpy as np
import meshcat
from gym.spaces import Box
from ncortex.utils import angle_diff
from .differentiable_env import DifferentiableEnv


class Pendulum(DifferentiableEnv):  #pylint: disable=too-many-instance-attributes
    '''
    A pendulum environment with a quadratic cost around the upright. The
    dynamics are integrated with forward Euler integration.
    '''

    def __init__( #pylint: disable=too-many-arguments
            self,
            x0=None,
            dt=0.01,
            g=9.81,
            R=None,
            Q=None,
            Q_f=None,
            zmq_url="tcp://127.0.0.1:6000"):

        # Create a null visualizer
        self._visualizer = None
        self.zmq_url = zmq_url

        # Define the size of the inputs and outputs.
        self.num_actuators = 1
        self.num_states = 2

        # Initialize the initial state.
        self.x0 = x0 if x0 is not None else np.array([0.0, 0.0])
        assert self.x0.shape[-1] == self.num_states

        # Dynamics Parameters
        self.g = g

        # Define cost terms.
        self.R = R if R is not None else dt * np.eye(
            self.num_actuators, dtype=np.float32)
        self.Q = Q if Q is not None else dt * np.eye(
            self.num_states, dtype=np.float32)
        self.Q_f = Q_f if Q_f is not None else np.eye(
            self.num_states, dtype=np.float32)
        self.goal = np.array([np.pi, 0.])

        # Define the action space.
        self.action_space = Box(
            np.array([-1]), np.array([1]), dtype=np.float32)

        super(Pendulum, self).__init__(x0=x0, dt=dt)

    @staticmethod
    def state_diff(state_1, state_2):
        ''' Compute the difference of two states and wrap angles properly.
        '''
        # Special case the vectorized version.
        if len(state_1.shape) < 2:
            theta_1 = state_1[:1]
            other_1 = state_1[1:]
        else:
            theta_1 = state_1[:, :1]
            other_1 = state_1[:, 1:]

        if len(state_2.shape) < 2:
            theta_2 = state_2[:1]
            other_2 = state_2[1:]
        else:
            theta_2 = state_2[:, :1]
            other_2 = state_2[:, 1:]

        # Subtract angles appropriately and everything else normally
        theta_diff = angle_diff(theta_1, theta_2)
        other_diff = other_1 - other_2

        if isinstance(theta_diff, tf.Tensor):
            return tf.concat([theta_diff, other_diff], axis=-1)

        return np.concatenate([theta_diff, other_diff], axis=-1)

    def transition_cost(self, state, action):
        ''' The cost of being in a state and taking an action.
        '''
        with tf.name_scope('cost'):
            err = self.state_diff(state, self.goal)
            if isinstance(state, tf.Tensor):
                state_cost = tf.reduce_sum(
                    tf.tensordot(err, self.Q, axes=[[-1], [0]]) * err, axis=-1)
                action_cost = tf.reduce_sum(
                    tf.tensordot(action, self.R, axes=[[-1], [0]]) * action,
                    axis=-1)
            else:
                state_cost = np.sum(
                    np.tensordot(err, self.Q, axes=[[-1], [0]]) * err, axis=-1)
                action_cost = np.sum(
                    np.tensordot(action, self.R, axes=[[-1], [0]]) * action,
                    axis=-1)

        return state_cost + action_cost

    def final_cost(self, state):
        ''' The cost of ending the simulation in a particular state.
        '''
        err = self.state_diff(state, self.goal)
        if isinstance(state, tf.Tensor):
            return tf.reduce_sum(
                tf.tensordot(err, self.Q_f, axes=[[-1], [0]]) * err, axis=-1)

        return np.sum(
            np.tensordot(err, self.Q_f, axes=[[-1], [0]]) * err, axis=-1)

    def reset(self):
        ''' Reset the pendulum to the zero state
        '''
        self.state = self.x0
        return self.state

    def dynamics(self, action):
        ''' Computes the state derivative.
        '''

        # Special case the vectorized version
        if len(self.state.shape) < 2:
            q = self.state[:1]
            dq = self.state[1:]
        else:
            q = self.state[:, :1]
            dq = self.state[:, 1:]

        if isinstance(q, tf.Tensor):
            d2q = -self.g*tf.sin(q) + action
            return tf.concat([dq, d2q], axis=-1)

        d2q = -self.g*np.sin(q) + action
        return np.concatenate([dq, d2q], axis=-1)

    @property
    def visualizer(self):
        '''
        Visualizer property. Initializes the visualizer if it hasn't been
        initialized yet.
        '''
        if self._visualizer is None:
            self._visualizer = meshcat.Visualizer(zmq_url=self.zmq_url)
            self._visualizer.open()
            self._visualizer["pendulum"].set_object(
                meshcat.geometry.Box([0.05, 0.05, 1.0]))
        return self._visualizer

    def render(self):
        '''
        Render the state of the environment. A tensorflow session must be open
        to evaluate the state.
        '''
        assert len(self.state.shape) == 1, \
            "Cannot render a vectorized Pendulum environment"
        if isinstance(self.state, tf.Tensor):
            theta = self.state[0].eval()
        else:
            theta = self.state[0]
        self.visualizer["pendulum"].set_transform(
            meshcat.transformations.rotation_matrix(theta, [1, 0, 0]).dot(
                meshcat.transformations.translation_matrix([0, 0, -.5])))
