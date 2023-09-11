import numpy as np

from jax import numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoDiagonalNormal

from numpyro.infer import MCMC, NUTS


def outcome_model(W_obs, mask, X, Y=None, mu=50):
    n = mask.size

    W_obs = numpyro.sample(
        "W_obs",
        dist.Uniform(0, 100).expand([n]).mask(mask),
        obs=W_obs,
    )
    W_latent = numpyro.sample(
        "W_impute",
        dist.Uniform(0, 100).expand([n]).mask(~mask),
    )

    W = W_obs * mask + W_latent * (~mask)
    Y_mu = jnp.dot(W, X)
    #numpyro.deterministic("Y", Y_mu, obs=Y)
    numpyro.sample("Y", dist.Normal(Y_mu, jnp.ones(1)), obs=Y)

def duel_model(W_obs, mask, X, Y=None):
    pass

if __name__ == "__main__":
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

    # bayesopt approach
    # later: move this to spdod/plan/model.py
    mask = masks[0].astype(bool)
    #w_obs = values[mask]
    w_obs = (values * mask)

    optimizer = numpyro.optim.Minimize()
    #optimizer = numpyro.optim.Adam(step_size=0.005, b1=0.5)
    guide = AutoLaplaceApproximation(outcome_model)
    #guide = AutoDiagonalNormal(outcome_model)
    svi = SVI(outcome_model, guide, optimizer, loss=Trace_ELBO())

    x_init = lsa(w_obs, maximize=True)
    x = np.zeros_like(values)
    x[x_init] = 1
    y = np.array([(values * x).sum()])

    init_state = svi.init(random.PRNGKey(0), w_obs.flatten(), mask.flatten(), x.flatten(), y)
    optimal_state, loss = svi.update(init_state, w_obs.flatten(), mask.flatten(), x.flatten(), y)
    params = svi.get_params(optimal_state)
    posterior = guide.get_posterior(params)
    print(posterior.mean)
    print(posterior.variance)


    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    # Run NUTS.
    kernel = NUTS(outcome_model)
    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X=x.flatten(),
        Y=y,
    )
    mcmc.print_summary()
    import pdb; pdb.set_trace()
