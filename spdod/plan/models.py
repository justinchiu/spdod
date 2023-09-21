# TODO: Implement BayesOpt[https://num.pyro.ai/en/stable/examples/thompson_sampling.html]
# with EI and KG acquisition functions
 
from scipy.optimize import linear_sum_assignment as lsa
import numpy as np

from jax import numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist

#from numpyro.infer import SVI, Trace_ELBO
#from numpyro.infer.autoguide import AutoLaplaceApproximation, AutoDiagonalNormal

from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive


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
    #Y_mu = jnp.dot(W, X)
    Y_mu = (W * X).sum(-1)
    #numpyro.deterministic("Y", Y_mu, obs=Y)
    numpyro.sample("Y", dist.Normal(Y_mu, jnp.ones(1)), obs=Y)

def duel_model(W_obs, mask, X1, X2, Y=None, mu=50):
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
    #Y_mu = jnp.dot(W, X)
    Y1_mu = (W * X1).sum(-1)
    Y2_mu = (W * X2).sum(-1)
    #numpyro.deterministic("Y", Y_mu, obs=Y)
    numpyro.sample("Y", dist.Bernoulli(logits=[Y_mu1, Y_mu2]), obs=Y)


def run_mcmc(rng_key, model, w_obs, mask, x, y):
    rng_key, rng_key_ = random.split(rng_key)

    # Run NUTS.
    kernel = NUTS(model)
    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
    mcmc.run(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X=x.reshape(-1, 64),
        Y=y,
    )
    #mcmc.print_summary()
    samples = mcmc.get_samples()

    #posterior_mu = samples["W_latent"].mean(0).reshape(values.shape)
    #posterior_std = samples["W_latent"].std(0).reshape(values.shape)
    return rng_key, samples

def ei(rng_key, model, samples, w_obs, mask, values):
    rng_key, rng_key_ = random.split(rng_key)
    batch_size = 64
    noise = dist.Gumbel(0,1).expand([batch_size,8,8]).sample(rng_key_)

    assns = [lsa(xs, maximize=True) for xs in noise]
    for i, assn in enumerate(assns):
        xs = np.zeros_like(values)
        xs[assn] = 1
        assns[i] = xs
    assns = jnp.stack(assns)

    rng_key, rng_key_ = random.split(rng_key)
    predictive = Predictive(model, samples)
    predictions = predictive(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X=assns.reshape(batch_size, 64),
    )
    predicted_Y = predictions["Y"].mean(0)

    next_x = assns[predicted_Y.argmax()]
    next_y = np.array([(values * next_x).sum()])
    return rng_key, next_x, next_y


def ei_2step(values, mask):
    # bayesopt approach
    #w_obs = values[mask]
    w_obs = (values * mask)


    x_init = lsa(w_obs, maximize=True)
    x = np.zeros_like(values)
    x[x_init] = 1
    y = np.array([(values * x).sum()])

    """ SVI stuff that i'm not using
    optimizer = numpyro.optim.Minimize()
    #optimizer = numpyro.optim.Adam(step_size=0.005, b1=0.5)
    guide = AutoLaplaceApproximation(outcome_model)
    #guide = AutoDiagonalNormal(outcome_model)
    svi = SVI(outcome_model, guide, optimizer, loss=Trace_ELBO())
    init_state = svi.init(random.PRNGKey(0), w_obs.flatten(), mask.flatten(), x.flatten(), y)
    optimal_state, loss = svi.update(init_state, w_obs.flatten(), mask.flatten(), x.flatten(), y)
    params = svi.get_params(optimal_state)
    posterior = guide.get_posterior(params)
    print(posterior.mean)
    print(posterior.variance)
    """


    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.PRNGKey(0)

    rng_key, samples = run_mcmc(rng_key, outcome_model, w_obs, mask, x, y)

    rng_key, next_x, next_y = ei(rng_key, outcome_model, samples, w_obs, mask, values)

    # next step
    rng_key, samples_2 = run_mcmc(
        rng_key,
        outcome_model,
        w_obs,
        mask,
        jnp.stack([x, next_x]),
        jnp.stack([y, next_y]),
    )

    rng_key, next_x2, next_y2 = ei(rng_key, outcome_model, samples, w_obs, mask, values)

    xs = np.array([x, next_x, next_x2])
    ys = np.array([y, next_y, next_y2])
    return xs, ys
    return xs[ys.argmax()]



if __name__ == "__main__":
    from dialop.envs import OptimizationEnv

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
    mask = masks[0].astype(bool)

    xs, ys = ei_2step(values, mask)

