
from dialop.envs import OptimizationEnv
from scipy.optimize import linear_sum_assignment as lsa

env = OptimizationEnv()
env.reset()
game = env.game

# ground truth
table = game.table
masks = game.masks
scales = game.scales
values = table.values

best_reward = game.best_assignment_reward

# observed
tables = game.tables


def score(table, values):
    rows, cols = lsa(values)
    assignment = list(zip(rows, cols))
    score = table.score(assignment)
    return score

import pdb; pdb.set_trace()
