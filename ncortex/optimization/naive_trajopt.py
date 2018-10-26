''' Naively optimize a trajectory of inputs.
'''

import tensorflow as tf


def run_naive_trajopt(env, x_0, u_init):
    '''
    Runs a naive trajectory optimization using built-in TensorFlow optimizers.
    '''

    # Validate u_init shape.
    N = u_init.shape[0]
    assert u_init.shape == (N, env.get_num_actuators)

    # Initialize the control variable.
    u = tf.Variable(u_init)

    # Compute the total reward
    env.state = x_0
    total_reward = tf.constant(0.)
    for i in range(N):
        _, reward, _, _ = env.step(u[i, :])
        total_reward += reward

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(total_reward)
    init = tf.global_variables_initializer()

    # Run the optimization procedure.
    with tf.Session() as sess:
        # Initialize all of the variables.
        sess.run(init)

        # Run the optimization procedure for 100 steps.
        for i in range(100):
            _, loss_value = sess.run((train, reward))
            print(loss_value)
