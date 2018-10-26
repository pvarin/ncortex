''' Implementation of the Iterated Linear Quadratic Regulator (iLQR) Algorithm
'''

# import tensorflow as tf

def run_ilqr(env, x_0, u_init): #pylint: disable=unused-argument
    ''' Runs the iLQR algorithm on an environment.
    '''
    # TODO: implement iLQR

    # # Validate u_init shape.
    # N = u_init.shape[0]
    # assert u_init.shape == (N, env.get_num_actuators)

    # # Initialize the control variable.
    # u = tf.Variable(u_init)

    # # Compute the total reward
    # env.state = x_0
    # total_reward = tf.constant(0.)
    # for i in range(N):
    #     _, reward, _, _ = env.step(u[i, :])
    #     total_reward += reward

    raise NotImplementedError
