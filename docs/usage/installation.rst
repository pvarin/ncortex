Installation and Quickstart
===========================


Installation
------------

1. Clone the repository from GitHub::

    git clone https://github.com/pvarin/ncortex

2. Install locally with ``pip``::

    cd ncortex
    pip install -e . 

Quickstart
-----------

1. Start the visualizer in a different terminal::

    meshcat-server

2. Simulate the pendulum environment::

    cd ncortex/examples
    python run_pendulum_env.py

Running Tests
--------------

We use the ``pytest`` framework for all of the tests and ``pylint`` for linting. By default the test command runs both the unit tests and the lint tests. You can run these tests with the command::

    python setup.py test
