import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

# import agent_student, expl_student, jaxutils_student
from . import agent_student_bisim, expl_student, jaxutils_student
from . import ninjax as nj


class Greedy(nj.Module):
    def __init__(self, wm, obs_space, act_space, teacher_wm, teacher_policy, config):
        self.teacher_wm = teacher_wm
        self.teacher_policy = teacher_policy
        rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
        if config.critic_type == "vfunction":
            critics = {"extr": agent_student_bisim.VFunction(rewfn, config, name="critic")}
        else:
            raise NotImplementedError(config.critic_type)
        self.ac = agent_student_bisim.ImagActorCritic(critics, {"extr": 1.0}, obs_space, act_space, teacher_wm, teacher_policy,  config, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)
    
    #########################################
    # Change 11/4/2025 19:46 PM
    #########################################

    def train(self, imagine, start, start_t, data):
        return self.ac.train(imagine, start, start_t, data , self.teacher_wm, self.teacher_policy)
    
    # def train(self, imagine, start, teacher_start, data):
    #     return self.ac.train(imagine, start, teacher_start, data, self.teacher_wm, self.teacher_policy)
    
    #########################################
    # End Change 11/2/2025 2:50 PM
    #########################################

    def report(self, data):
        return {}


class Random(nj.Module):
    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return jnp.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = jaxutils_student.OneHotDist(jnp.zeros(shape))
        else:
            dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
            dist = tfd.Independent(dist, 1)
        return {"action": dist}, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}


class Explore(nj.Module):
    REWARDS = {
        "disag": expl_student.Disag,
    }

    def __init__(self, wm, act_space, config):
        self.config = config
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            if key == "extr":
                rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
                critics[key] = agent_student_bisim.VFunction(rewfn, config, name=key)
            else:
                rewfn = self.REWARDS[key](wm, act_space, config, name=key + "_reward")
                critics[key] = agent_student_bisim.VFunction(rewfn, config, name=key)
                self.rewards[key] = rewfn
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = agent_student_bisim.ImagActorCritic(critics, scales, act_space, config, name="ac")

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        metrics = {}
        for key, rewfn in self.rewards.items():
            mets = rewfn.train(data)
            metrics.update({f"{key}_k": v for k, v in mets.items()})
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}
    
