''' Implementation of the Iterated Linear Quadratic Regulator (iLQR) Algorithm
'''

def run_ilqr(env, x_0, u_init):
    ''' Runs the iLQR algorithm on an environment.
    '''

    # Valitate u_init shape.
    N = u_init.shape[0]
    assert u_init.shape == (N, env.get_num_actuators)

    # Forward Pass
    x = x_0
    u = u_init
    for i in range(N):
        x = env.step(x, u[i, :])

    raise NotImplementedError
