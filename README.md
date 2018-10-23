# ncortex
Ncortex, short for neocortex, is a repository for both model-based and model-free reinforcement learning implementations, with a specific focus on robotics.

The focus of this project is to provide:
- Good implementations of model-based and model-free RL algorithms
- Support for learning on differentiable and non-differentiable dynamics
- Low activation-cost environment for RL experimentation
- Good, complete diagnostic tools, specifically tensorboard logging

## Installation
Install locally by running
```
python setup.py develop
```
from the root directory.

## Testing
Run all of the test cases with 
```
pytest --pylint --pyargs ncortex
```
from the root directory.