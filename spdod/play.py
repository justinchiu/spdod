
from dialop.envs import OptimizationEnv
from scipy.optimize import linear_sum_assignment as lsa
import seaborn as sns
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib

from spdod.plan.models import ei_2step

NUM_TRIALS = 100

env = OptimizationEnv()

# dont use
max_ratios = []

both_ratios = []
ei_both_ratios = []
times = []
ei_times = []
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

    # bayesopt approach
    # later: move this to spdod/plan/model.py
    mask = masks[0].astype(bool)
    w_known = values[mask]

    # what we will use for guessing
    W = np.zeros_like(values)
    W[mask] = w_known

    # create latent variable
    num_latents = W.size - mask.sum()
    w = np.full(num_latents, 25)

    # guess
    x = lsa(W, maximize=True)
    # value
    y = values[x].sum()

    start = time.time()
    xs, ys = ei_2step(values, mask)
    value_ei = ys.max()
    ei_times.append(time.time() - start)

    def score(table, values):
        rows, cols = lsa(values)
        assignment = list(zip(rows, cols))
        score = table.score(assignment)
        return score

    start = time.time()
    max_value = values[lsa(values, maximize=True)].sum()
    value0 = values[lsa(values * masks[0], maximize=True)].sum()
    value_both = values[lsa(values * (masks[0] | masks[1]), maximize=True)].sum()
    times.append(time.time() - start)

    print(n)
    print(value0 / value_both)
    #print(value0 / max_value)
    print(value_ei / value_both)
    both_ratios.append(value0 / value_both)
    max_ratios.append(value0 / max_value)

    ei_both_ratios.append(value_ei / value_both)
    #import pdb; pdb.set_trace()
    
#df = pd.DataFrame({"Max ratio": max_ratios, "Both ratio": both_ratios})
df = pd.DataFrame({"EI both ratio": ei_both_ratios, "Both ratio": both_ratios})

# Create violin plot
sns.violinplot(data=df)

# Save the plot
plt.savefig("figures/max_both_ratios.png")
print("Time per solve:", np.array(times).mean() / 3)
print("Time per EI MCMC solve:", np.array(ei_times).mean())
