''' Pendulum class.
'''
import tensorflow as tf
import autograd.numpy as np
import meshcat
from gym.spaces import Box
from .differentiable_env import DifferentiableEnv
from .costs import quadratic_cost


class DoubleIntegrator(DifferentiableEnv):  #pylint: disable=too-many-instance-attributes
    '''
    A pendulum environment with a quadratic cost around the upright. The
    dynamics are integrated with forward Euler integration.
    '''

    def __init__(  #pylint: disable=too-many-arguments
            self,
            x_0=None,
            dt=0.01,
            R=None,
            Q=None,
            Q_f=None,
            dtype=None,
            zmq_url="tcp://127.0.0.1:6000",
            use_tf=True):

        # Choose the correct numerical library and data type
        self.use_tf = use_tf
        if dtype is not None:
            self.dtype = dtype
        elif use_tf:
            self.dtype = tf.float32
        else:
            self.dtype = np.float64

        # Create a null visualizer
        self._visualizer = None
        self.zmq_url = zmq_url

        # Define the size of the inputs and outputs.
        self.num_actuators = 1
        self.num_states = 2

        # Initialize the initial state.
        if self.use_tf:
            self.x_0 = x_0 if x_0 is not None else tf.constant([0.0, 0.0],
                                                               dtype=dtype)
        else:
            self.x_0 = x_0 if x_0 is not None else np.array([0.0, 0.0],
                                                            dtype=dtype)
            assert self.x_0.shape[-1] == self.num_states

        # Define cost terms.
        if self.use_tf:
            self.R = R if R is not None else dt * tf.eye(
                self.num_actuators, dtype=self.dtype)
            self.Q = Q if Q is not None else dt * tf.eye(
                self.num_states, dtype=self.dtype)
            self.Q_f = Q_f if Q_f is not None else tf.eye(
                self.num_states, dtype=self.dtype)
            self.goal = tf.constant([0, 0.])
        else:
            self.R = R if R is not None else dt * np.eye(
                self.num_actuators, dtype=self.dtype)
            self.Q = Q if Q is not None else dt * np.eye(
                self.num_states, dtype=self.dtype)
            self.Q_f = Q_f if Q_f is not None else np.eye(
                self.num_states, dtype=self.dtype)
            self.goal = np.array([0, 0.])

        # Define the action space.
        if use_tf:
            self.action_space = Box(
                np.array([-1]),
                np.array([1]),
                dtype=self.dtype.as_numpy_dtype())
        else:
            self.action_space = Box(
                np.array([-1]), np.array([1]), dtype=self.dtype)

        super(DoubleIntegrator, self).__init__(dt=dt)

    def transition_cost(self, state, action):
        ''' The cost of being in a state and taking an action.
        '''
        err = state - self.goal
        state_cost = quadratic_cost(err, self.Q, self.use_tf)
        action_cost = quadratic_cost(action, self.R, self.use_tf)
        return state_cost + action_cost

    def final_cost(self, state):
        ''' The cost of ending the simulation in a particular state.
        '''
        err = state - self.goal
        if self.use_tf:
            # Check for vectorized environments since tf.einsum() doesn't
            #   support ellipses yet
            if len(state.get_shape()) == 1:
                return tf.einsum('i,ij,j', err, self.Q_f, err)
            return tf.einsum('ij,jk,ik->i', err, self.Q_f, err)

        return np.einsum('...i,ij,...j->...', err, self.Q_f, err)

    def reset(self):
        ''' Reset the pendulum to the zero state
        '''
        return self.x_0

    def dynamics(self, state, action):
        ''' Computes the state derivative.
        '''

        # Special case the vectorized version
        if len(state.shape) < 2:
            dq = state[1:]
        else:
            dq = state[:, 1:]

        d2q = action

        if self.use_tf:
            return tf.concat([dq, d2q], axis=-1)

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
                meshcat.geometry.Box([0.1, 0.1, 0.1]))
        return self._visualizer

    def render(self, state):
        '''
        Render the state of the environment. A tensorflow session must be open
        to evaluate the state.
        '''
        assert len(state.shape) == 1, "Cannot render a vectorized environment"
        if self.use_tf:
            pos = state[0].eval()
        else:
            pos = state[0]
        self.visualizer["pendulum"].set_transform(
            meshcat.transformations.translation_matrix([0, pos, 0]))
