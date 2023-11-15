# TODO: Implement BayesOpt[https://num.pyro.ai/en/stable/examples/thompson_sampling.html]
# with EI and KG acquisition functions
 
from scipy.optimize import linear_sum_assignment as lsa
import numpy as np

from jax import numpy as jnp
from jax import random
from jax.scipy.special import logsumexp

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
        "W_latent",
        dist.Uniform(0, 100).expand([n]).mask(~mask),
    )

    W = W_obs * mask + W_latent * (~mask)
    #Y_mu = jnp.dot(W, X)
    Y_mu = (W * X).sum(-1)
    #numpyro.deterministic("Y", Y_mu, obs=Y)
    numpyro.sample("Y", dist.Normal(Y_mu, jnp.ones(1)), obs=Y)

def duel_model(W_obs, mask, X1=None, X2=None, Y=None, mu=50):
    n = mask.size

    W_obs = numpyro.sample(
        "W_obs",
        dist.Uniform(0, 100).expand([n]).mask(mask),
        obs=W_obs,
    )
    W_latent = numpyro.sample(
        "W_latent",
        dist.Uniform(0, 100).expand([n]).mask(~mask),
    )

    W = W_obs * mask + W_latent * (~mask)
    if X1 is not None and X2 is not None:
        #Y_mu = jnp.dot(W, X)
        Y_mu1 = (W[X1]).sum(-1)
        Y_mu2 = (W[X2]).sum(-1)
        #numpyro.deterministic("Y", Y_mu, obs=Y)
        temperature = 1
        logits = jnp.array([Y_mu1, Y_mu2]) / temperature
        Z = logsumexp(logits)
        numpyro.sample("Y", dist.Bernoulli(logits=Y_mu1 - Z), obs=Y)


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
        X=x.reshape(-1, 64) if x is not None else x,
        Y=y,
    )
    #mcmc.print_summary()
    samples = mcmc.get_samples()

    #posterior_mu = samples["W_latent"].mean(0).reshape(values.shape)
    #posterior_std = samples["W_latent"].std(0).reshape(values.shape)
    return rng_key, samples

def run_duel_mcmc(rng_key, model, w_obs, mask, x1, x2, y):
    rng_key, rng_key_ = random.split(rng_key)

    # Run NUTS.
    kernel = NUTS(model)
    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=False)
    mcmc.run(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X1=x1.reshape(-1, 64) if x1 is not None else x1,
        X2=x2.reshape(-1, 64) if x2 is not None else x2,
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


def ei_2step(rng_key, values, mask):
    # bayesopt approach
    #w_obs = values[mask]
    w_obs = (values * mask)

    x_init = lsa(w_obs, maximize=True)
    x = np.zeros_like(values)
    x[x_init] = 1
    y = np.array([(values * x).sum()])

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

    rng_key, next_x2, next_y2 = ei(rng_key, outcome_model, samples_2, w_obs, mask, values)

    xs = np.array([x, next_x, next_x2])
    ys = np.array([y, next_y, next_y2])
    return xs, ys


def kg(rng_key, model, samples, w_obs, mask, values):
    rng_key, rng_key_ = random.split(rng_key)
    batch_size = 64
    noise = dist.Gumbel(0,1).expand([3, batch_size,8,8]).sample(rng_key_)

    # sample uniformly random full assignments
    # TODO: sample assignments from the prior
    assns = [lsa(xs, maximize=True) for xs in noise[0]]
    for i, assn in enumerate(assns):
        xs = np.zeros_like(values)
        xs[assn] = 1
        assns[i] = xs
    assns = jnp.stack(assns)

    # sample pairwise comparisons
    pairs = noise[1:,:]
    rng_key, rng_key_ = random.split(rng_key)
    predictive = Predictive(model, samples)
    predictions = predictive(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X=assns.reshape(batch_size, 64),
    )
    predicted_W = predictions["W_latent"].mean(0)
    import pdb; pdb.set_trace()

    next_x = assns[predicted_Y.argmax()]
    next_y = np.array([(values * next_x).sum()])
    return rng_key, next_x, next_y

def kg_2step(rng_key, values, mask):
    w_obs = (values * mask)
    x_init = lsa(w_obs, maximize=True)
    x = np.zeros_like(values)
    x[x_init] = 1
    y = np.array([(values * x).sum()])


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
    w_obs = values * mask

    #xs, ys = ei_2step(values, mask)

    # messing with KG
    # Steps in KG:
    # 1. sample candidate actions
    # 2. for each candidate action: compute marginal weight posterior (marginalizing over responses)
    # 3. for each candidate action: compute expected utility from marginal weight posterior mean
    # 4. pick action with highest expected utility
     
    rng_key = random.PRNGKey(0)


    rng_key, rng_key_ = random.split(rng_key)
    rng_key, samples = run_duel_mcmc(rng_key_, duel_model, w_obs, mask, x1=None, x2=None, y=None)

    prior_mean = w_obs + samples["W_latent"].mean(0).reshape(8,8) * ~mask
    rng_key, rng_key_ = random.split(rng_key)
    noise = dist.Gumbel(0,1).expand([128,8,8]).sample(rng_key_)
    assns = [lsa(prior_mean + xs, maximize=True) for xs in noise]
    x1 = np.stack(assns[:len(assns)//2])
    x2 = np.stack(assns[len(assns)//2:])
    x1 = (x1[:,0] * 8 + x1[:,1])
    x2 = (x2[:,0] * 8 + x2[:,1])
    #x0 = lsa(prior_mean, maximize=True)
    #y0 = prior_mean[x0].sum()

    rng_key, rng_key_ = random.split(rng_key)
    predictive = Predictive(duel_model, samples)
    predictions = predictive(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X1=x1,
        X2=x2,
        Y=None,
    )
    p_y = predictions["Y"].mean(0)
    posterior_predictions = predictive(
        rng_key_,
        W_obs=w_obs.flatten(),
        mask=mask.flatten(),
        X1=x1,
        X2=x2,
        Y=None,
    )


    import pdb; pdb.set_trace()
