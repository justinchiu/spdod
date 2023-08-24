
from dialop.envs import OptimizationEnv
from scipy.optimize import linear_sum_assignment as lsa
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib


NUM_TRIALS = 100

env = OptimizationEnv()

both_ratios = []
max_ratios = []
for n in range(NUM_TRIALS):
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

    max_value = values[lsa(values, maximize=True)].sum()
    value0 = values[lsa(values * masks[0], maximize=True)].sum()
    value_both = values[lsa(values * (masks[0] | masks[1]), maximize=True)].sum()

    print(n)
    print(value0 / value_both)
    print(value0 / max_value)
    both_ratios.append(value0 / value_both)
    max_ratios.append(value0 / max_value)
    #import pdb; pdb.set_trace()
    
df = pd.DataFrame({"Max ratio": max_ratios, "Both ratio": both_ratios})

# Create violin plot
sns.violinplot(data=df)

# Save the plot
plt.savefig("figures/max_both_ratios.png")
