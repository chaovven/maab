from .q_learner import QLearner
from .q_learner_v import QLearnerV
from .q_learner_gan import QLearnerGan

REGISTRY = {}

REGISTRY["q_learner_v"] = QLearnerV
REGISTRY["q_learner_gan"] = QLearnerGan
REGISTRY["q_learner"] = QLearner
