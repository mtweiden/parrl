# ParRL Library
This module provides boilerplate code for training RL systems in parallel with
ray. ParRL is a PyTorch focused modular library for RL projects.

## Architecture
The system architecture depends on two entities, the Learner and the Gatherer.
1. Learners
  Learners are responsible for agent training and for maintaining Gatherers. The 
  Learner interacts with environment experience through the use of a ReplayBuffer.
  Typically, a Learner will host multiple Gatherers.

2. Gatherers
  Gatherers are responsible for gathering experiences that can be added to the Learner's ReplayBuffer. A Gatherer is a Ray Actor that houses a copy of the
  Learner's agent and an Environment.