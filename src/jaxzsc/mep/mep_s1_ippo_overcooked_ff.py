""" 
Based on PureJaxRL Implementation of PPO
"""
import datetime
import os
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
import wandb
import pyrallis
import jaxmarl

from typing import Literal, Sequence, NamedTuple, Any
from dataclasses import asdict, dataclass

from flax import core, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts


@dataclass
class TrainConfig:
    # Wandb and other logging
    project: str = "JaxZSC"
    mode: Literal["online", "offline", "disabled"] = "disabled"
    group: str = ""
    entity: str = ""
    checkpoint_path: str = "checkpoints"
    checkpoint_freq: int = 100 # Checkpoint every N updates
    # MEP
    population_size: int = 2
    ent_pop_coeff: float = 0.01

    # Overcooked
    layout_name: Literal["cramped_room", "asymm_advantages", "coord_ring", "forced_coord", "counter_circuit"] = "cramped_room"
    rew_shaping_horizon: int = 1e7

    # Actor-Critic
    activation: str = "tanh"

    # Training
    seed: int = 42
    lr: float = 2.5e-4
    anneal_lr: bool = True
    num_envs: int = 128
    num_steps: int = 400
    total_timesteps: int = 1e8
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    def __post_init__(self):
        self.num_actors = 2 * self.num_envs
        self.num_updates = int(self.total_timesteps // self.num_steps // self.num_envs)
        self.minibatch_size = self.num_actors * self.num_steps // self.num_minibatches

        print("Number of updates: ", self.num_updates)


class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)


def rollout(rng, layout_name, activation_string, params) -> RolloutStats:
    def _cond_fn(carry):
        rng, env_state, stats, obsv, done = carry
        return (done != True).any() # Continue if not done.

    def _body_fn(carry):
        rng, env_state, stats, last_obs, done = carry

        rng, rng_action, rng_step = jax.random.split(rng, 3)
        obs_batch = batchify(last_obs, env.agents, 2)
        pi, _ = network.apply(params, obs_batch)
        action = pi.sample(seed=rng_action).squeeze()

        env_act = unbatchify(action, env.agents, 1, env.num_agents)
        env_act = {k: v.flatten().squeeze() for k,v in env_act.items()}

        obsv, env_state, reward, done, info = env.step(
            rng_step, env_state, env_act
        )

        stats = stats.replace(
            reward=stats.reward + reward["agent_0"],
            length=stats.length + 1
        )
        carry = (rng, env_state, stats, obsv, done["__all__"])
        return carry
    
    key, key_r = jax.random.split(rng)
    env = jaxmarl.make("overcooked", layout=overcooked_layouts[layout_name])
    network = ActorCritic(env.action_space().n, activation_string)
    obs, state = env.reset(key_r)

    init_carry = (rng, state, RolloutStats(), obs, jnp.array(False))

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[2].reward.squeeze(), final_carry[2].length.squeeze()


class PopulationTrainState(TrainState):

    population: core.FrozenDict[str, Any]
    other_agent_idcs: jnp.ndarray
    curr_agent_idx: int


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    neg_logp_pop_new: jnp.ndarray
    orig_reward: jnp.ndarray
    shaped_reward: jnp.ndarray
    entropy_pop_delta: jnp.ndarray
    neg_logp_pop_delta: jnp.ndarray


MAX_ENT = -jnp.log(1/6)
def entropy(action_probs):
    # assert action_probs.shape[1] == 6, 'action_probs.shape[1] == 6'
    neg_p_logp = - action_probs * jnp.log(action_probs)
    entropy = jnp.sum(neg_p_logp, axis=1)
    # assert jnp.max(entropy) <= MAX_ENT+1e5, 'entropy_max <= MAX_ENT'
    return entropy


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_update_fn(config, env, network):
    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config.rew_shaping_horizon
    )

        # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
            obs_batch = obs_batch.reshape(config.num_actors, -1)

            pi, value = network.apply(train_state.params, obs_batch)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # --------------------- #
            # NOTE: THIS IS FOR MEP #
            action_probs_agent0 = pi.probs
            actions = action
            # --------------------- #

            action_probs_np = jnp.zeros((config.population_size, config.num_actors, 6)) ## 6 is the action_dim
            actions_np = jnp.zeros((config.population_size, config.num_actors)) # Is 128 the batch size? The number of parallel envs?

            for i in range(config.population_size):
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.population[i], obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                actions_np = actions_np.at[i].set(action)
                action_probs_np = action_probs_np.at[i].set(pi.probs)

            # Remove current agent
            action_probs_np = jnp.delete(action_probs_np, train_state.curr_agent_idx, axis=0, assume_unique_indices=True)
            actions_np = jnp.delete(actions_np, train_state.curr_agent_idx, axis=0, assume_unique_indices=True)
            
            action_probs_pop_np = jnp.mean(action_probs_np, axis=0)

            action_probs_np_new = jnp.append(action_probs_np, jnp.expand_dims(action_probs_agent0, axis=0), axis=0)

            action_probs_pop_np_new = jnp.mean(action_probs_np_new, axis=0)
            entropy_pop = entropy(action_probs_pop_np)
            entropy_pop_new = entropy(action_probs_pop_np_new)
            entropy_pop_delta = entropy_pop_new - entropy_pop

            sampled_action_prob_pop_np = jnp.take(action_probs_pop_np, actions)
            neg_logp_pop = - jnp.log(sampled_action_prob_pop_np)
            sampled_action_prob_pop_np_new = jnp.take(action_probs_pop_np_new, actions)
            neg_logp_pop_new = - jnp.log(sampled_action_prob_pop_np_new)
            neg_logp_pop_delta = neg_logp_pop_new - neg_logp_pop


            env_act = unbatchify(
                action, env.agents, config.num_envs, env.num_agents
            )

            env_act = {k: v.flatten() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config.num_envs)

            # ipdb.set_trace()
            obsv, env_state, orig_reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)

            shaped_reward = info.pop("shaped_reward")
            current_timestep = update_step*config.num_steps*config.num_envs
            reward = jax.tree.map(lambda x,y: x+y*rew_shaping_anneal(current_timestep), orig_reward, shaped_reward)
            neg_logp_pop_new_a = neg_logp_pop_new.reshape((2, config.num_envs))
            neg_logp_pop_new_a = {"agent_0": neg_logp_pop_new[0], "agent_1": neg_logp_pop_new[1]}
            reward = jax.tree.map(lambda x,y: x+y*config.ent_pop_coeff, reward, neg_logp_pop_new_a)

            info = jax.tree.map(lambda x: x.reshape((config.num_actors)), info)

            transition = Transition(
                batchify(done, env.agents, config.num_actors).squeeze(),
                action,
                value,
                batchify(reward, env.agents, config.num_actors).squeeze(),
                log_prob,
                obs_batch,
                info,
                neg_logp_pop_new,
                batchify(orig_reward, env.agents, config.num_actors).squeeze(),
                batchify(shaped_reward, env.agents, config.num_actors).squeeze(),
                entropy_pop_delta,
                neg_logp_pop_delta,
            )
            runner_state = (train_state, env_state, obsv, update_step, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config.num_steps
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, update_step, rng = runner_state
        last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
        last_obs_batch = last_obs_batch.reshape(config.num_actors, -1)

        _, last_val = network.apply(train_state.params, last_obs_batch)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config.gamma * next_value * (1 - done) - value
                gae = (
                    delta
                    + config.gamma * config.gae_lambda * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config.clip_eps, config.clip_eps)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.clip_eps,
                            1.0 + config.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config.vf_coef * value_loss
                        - config.ent_coef * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config.minibatch_size * config.num_minibatches
            assert (
                batch_size == config.num_steps * config.num_actors
            ), "batch size must be equal to number of steps * number of actors"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config.num_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.update_epochs
        )
        train_state = update_state[0]
        metric = traj_batch.info
        rng = update_state[-1]

        def callback(metric):
            wandb.log(metric)

        update_step = update_step + 1
        metric = jax.tree.map(lambda x: x.mean(), metric)
        metric["update_step"] = update_step
        metric["env_step"] = update_step * config.num_steps * config.num_envs
        metric["neg_logp_pop_new"] = jnp.mean(traj_batch.neg_logp_pop_new)
        metric["entropy_pop_delta"] = jnp.mean(traj_batch.entropy_pop_delta)
        metric["neg_logp_pop_delta"] = jnp.mean(traj_batch.neg_logp_pop_delta)
        metric["orig_reward"] = traj_batch.orig_reward.sum(axis=0).mean() / 2
        metric["shaped_reward"] = traj_batch.shaped_reward.sum(axis=0).mean()
        jax.debug.callback(callback, metric)

        runner_state = (train_state, env_state, last_obs, update_step, rng)
        return runner_state, metric
    return _update_step



def get_run_string(config: TrainConfig):
    return f"FF_MEP_IPPO_Overcooked_{config.layout_name}"


@pyrallis.wrap()
def train(config: TrainConfig):
    ##### WANDB and other setup #####
    tags = [
        "FF",
        "MEP",
        "IPPO",
        config.layout_name,
    ]
    run = wandb.init(
        project=config.project,
        group=config.group,
        mode=config.mode,
        config=asdict(config),
        save_code=True,
        tags=tags,
    )

    run_string = get_run_string(config)
    run.name = run.name + "___" + run_string

    #### Setup and check saving before training ####
    if config.checkpoint_path is not None:
        save_dir = os.path.join(config.checkpoint_path, run.name)
        # Make sure we can write the checkpoint later _before_ we wait 1 day for training!
        os.makedirs(save_dir, exist_ok=True)


    env = jaxmarl.make("overcooked", layout=overcooked_layouts[config.layout_name])
    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        return config.lr * frac
    
    rng = jax.random.PRNGKey(config.seed)

    # INIT NETWORK
    network = ActorCritic(env.action_space().n, activation=config.activation)
    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    # network parameter dictionary
    a_jnp_dict = {}
    for i in range(config.population_size):
        rng, _rng_a = jax.random.split(rng, 2)
        a_jnp_dict[i] = network.init(_rng_a, init_x) 


    if config.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config["LR"], eps=1e-5))

    key = jax.random.PRNGKey(0)
    all_agent_idcs = jnp.arange(config.population_size)
    param_idx = int(jax.random.choice(key, all_agent_idcs))
    other_agent_idcs = all_agent_idcs[all_agent_idcs != param_idx]
    params_ts = a_jnp_dict[param_idx]

    train_state = PopulationTrainState.create(
        apply_fn=network.apply,
        params=params_ts,
        tx=tx,
        population=a_jnp_dict,
        other_agent_idcs=other_agent_idcs,
        curr_agent_idx=param_idx,
    )

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn(config, env, network)
    jitted_update_step = jax.jit(_update_step)

    # INIT EVAL ROLLOUT FUNCTION
    jitted_rollout = jax.jit(rollout, static_argnums=(1,2)) # config is static
    
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    
    runner_state = (train_state, env_state, obsv, 0, _rng)

    now = '{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now())

    old_param_idx = param_idx    
    for i in range(config.num_updates):
        key, _key = jax.random.split(key)

        runner_state, metric = jitted_update_step(runner_state)
        train_state = runner_state[0]

        # Now we need to update the population with the newly trained agent
        new_param_idx = int(jax.random.choice(_key, all_agent_idcs))
        population = train_state.population
        new_params_to_train = population[new_param_idx]
        population[old_param_idx] = train_state.params # Update the newly trained agent in population with the new parameters
        train_state = train_state.replace(
            population=population,
            params=new_params_to_train,
            curr_agent_idx=new_param_idx,
            other_agent_idcs=all_agent_idcs[all_agent_idcs != new_param_idx]
        )
        
        runner_state = (train_state, ) + runner_state[1:]
        old_param_idx = new_param_idx


        # Remarkably, saving is among the most expensive operations
        if i % config.checkpoint_freq == 0 and i != 0:
            print(i)
            for p in range(config.population_size):
                params = train_state.population[p]
                total_r, total_l = jitted_rollout(rng, config.layout_name, config.activation, params)
                print(total_l)

                path = f"{save_dir}/{config.layout_name}/{now}/{p}" 
                os.makedirs(path, exist_ok=True)
                payload = (None, {"actor_params": params})
                pickle.dump(payload, open(path + f"/params_{i}_{total_r}.pt", "wb"))
                print("Saved params for agent", p, "with total reward", total_r)

 

    return {"runner_state": runner_state, "metrics": metric}

if __name__ == '__main__':
    train()
