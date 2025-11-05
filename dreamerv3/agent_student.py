import jax
import jax.numpy as jnp
import distrax

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors_student, jaxagent_student, jaxutils_student, nets_student
from . import ninjax as nj



@jaxagent_student.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, teacher_wm, teacher_policy, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.teacher_wm = teacher_wm
        self.teacher_policy = teacher_policy
        self.step = step
        self.start = None
        self.context = None
        # print(config)
        self.wm = WorldModel(obs_space, act_space, teacher_wm, teacher_policy, self.start, self.context, config, name="wm")
        # print("*********************************************************")
        # self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config, name="task_behavior")
        # if config.expl_behavior == "None":
        #     self.expl_behavior = self.task_behavior
        # else:
        #     self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config, name="expl_behavior")

        # 'task_behavior' is your standard Dreamer policy for environment reward
        self.task_behavior = behaviors_student.Greedy(self.wm, self.obs_space, self.act_space, teacher_wm, teacher_policy, self.config, name="task_behavior")

        # The "teacher_expl" is the module that trains RL with teacher reward
        # self.teacher_expl = behaviors_student.TeacherExpl(self.wm, act_space, config)
        # self.teacher_expl = behaviors_student.GreedyStudent(self.wm, self.obs_space, self.act_space, teacher_wm, teacher_policy, self.config, name="teacher_expl")


        # If you want the agent's final policy to be a mixture or pick one, up to you.
        # For example, you can do:
        self.expl_behavior = self.task_behavior



        def print_teacher_params_norm(agent, label):
            """Compute and print the norm of the teacher's world model params."""
            # 1) Collect submodules that hold parameters:
            wm = agent  # The teacher's world model
            submodules = [
                wm.encoder,
                wm.rssm,
                wm.heads["decoder"],
                wm.heads["reward"],
                wm.heads["cont"],
            ]
            
            # 2) Flatten them into leaves:
            leaves = jax.tree_util.tree_leaves(submodules)
            
            # 3) Filter out any that aren't jnp arrays
            array_leaves = [x for x in leaves if isinstance(x, jnp.ndarray)]
            
            # 4) Compute param norm
            if not array_leaves:
                print(f"[{label}] No param arrays found in teacher_wm submodules.")
                return
            param_norm = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in array_leaves))
            print(f"[{label}] Teacher param norm:", param_norm)

        # Print before training
        print_teacher_params_norm(teacher_wm, label="Before training")

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode="train"):
        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        state = ((latent, outs["action"]), task_state, expl_state)
        return outs, state

    def train(self, data, teacher_data, state, teacher_state,traj=None, teacher_traj=None):
        # print("**data:", data.keys()

        # print("1state:", state)  
        # print("1teacher_state:", teacher_state)
        self.config.jax.jit and print("Tracing train function.")
        # print("**data:", data.keys()

        # print("2state:", state)  
        # print("2teacher_state:", teacher_state)
        metrics = {}
        # print("**data:", data.keys()

        # print("3state:", state)  
        # print("3teacher_state:", teacher_state)
        data = self.preprocess(data)

        # print("**data:", data.keys()

        # print("4state:", state)  
        # print("4teacher_state:", teacher_state)
        teacher_data = self.preprocess(teacher_data)
        # print("**data:", data.keys()

        # print("state:", state)  
        # print("teacher_state:", teacher_state)




        state, teacher_state, wm_outs, mets = self.wm.train(data, teacher_data, state,teacher_state,traj, teacher_traj)
        metrics.update(mets)

        # 2) For the teacher reward RL
        context = {**data, **teacher_data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)


        #########################################
        # Change 11/4/2025 19:46 PM
        #########################################

        traj, teacher_traj, mets_expl = self.task_behavior.train(self.wm.imagine, start, context)
        # traj, teacher_traj, mets_expl = self.task_behavior.train(self.wm.imagine, self.wm.start, self.wm.teacher_start, context)

        #########################################
        # End Change 11/2/2025 2:50 PM
        #########################################

        # after: traj, teacher_traj, outs, state[0], teacher_state[0], mets = agent.train(...)

        def _is_distill_key(k: str) -> bool:
            # WorldModel: KLs and imagined dist terms are created as losses
            #   e.g. posterior_stoch_kl_loss_mean, prior_deter_kl_loss_std,
            #        dist_loss_imagined_loss_mean, dist_stoch_imagined_loss_std
            # Actor-side: we keep only the scalar KL summary
            return (
                k.startswith("posterior_") or
                k.startswith("prior_") or
                k.startswith("dist_") or
                k in ("kl_mean",  # actor’s teacher-vs-student KL summary
                    "teacher_wm_l2", "teacher_wm_l2_delta",
                    "teacher_actor_l2", "teacher_actor_l2_delta")  # if you added fingerprints
            )

        # distill_mets = {k: v for k, v in mets.items() if _is_distill_key(k)}
        # train_mets   = {k: v for k, v in mets.items() if k not in distill_mets}

        # metrics.update(train_mets,   prefix="train")
        # metrics.update(distill_mets, prefix="distill")




        metrics.update(mets_expl)

        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        if "keyA" in data.keys():
            outs = {
                "key": data["key"],
                "env_step": data["env_step"],
                "model_loss": metrics["model_loss_raw"].copy(),
                "td_error": metrics["td_error"].copy(),
            }

        else:
            outs = {}

        # Don't need the full model_loss_raw or td_error after the priority calculation, summarize it.
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        # teacher_params_after = nj.params(self.teacher_wm)
        # norm_after = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(teacher_params_after)))
        # print("Teacher param norm after train call:", norm_after)

        return traj, teacher_traj, outs, state, teacher_state, metrics




    def report(self, data,teacher_data):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        teacher_data = self.preprocess(teacher_data)
        report = {}
        report.update(self.wm.report(data,teacher_data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils_student.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        # if "teacher_action" in obs:
        #     pass
        # else:
        #     if"action" in obs:
        #         # print("deriving from action")
        #         # Assuming teacher_action is a discrete action, one-hot encoded
        #         # or continuous. Adjust the shape accordingly.
        #         obs["teacher_action"] = obs["action"].astype(jnp.float32)
        #     else:
        #         pass
        # # obs["teacher_action"] = obs["action"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, teacher_wm, teacher_policy, start, context, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.teacher_wm = teacher_wm
        self.teacher_policy = teacher_policy
        self.start = start
        self.context = context
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        # print("config.encoder:", **config.encoder)
        self.encoder = nets_student.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets_student.RSSM(**config.rssm, name="rssm")

        dec_shapes = {k: v for k, v in shapes.items() if k != "desired_goal"}

        self.heads = {
            "decoder": nets_student.MultiDecoder(dec_shapes, **config.decoder, name="dec"),
            # "teacher_action": nets_student.MLP(shape=act_space["action"].shape, **config.teacher_head, name="teacher_action"),
            "reward": nets_student.MLP((), **config.reward_head, name="rew"),
            "cont": nets_student.MLP((), **config.cont_head, name="cont"),
            
        }

        # print(act_space)

        # self.actor = nets_student.MLP(
        #     name="actor",
        #     dims="deter",
        #     shape=act_space["action"].shape,
        #     **config.actor,
        #     dist=config.actor_dist_disc,
        # )

        # self.policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())


        self.opt = jaxutils_student.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action
        

    def train(self, data, teacher_data, state,teacher_state,traj=None, teacher_traj=None):
        # self.traj = traj
        # self.teacher_traj = teacher_traj
        print("wmstate:", state)
        print("wmteacher_state:", teacher_state)

        modules = [self.encoder, self.rssm, *self.heads.values()]

        print("**data:", data.keys())
        mets, (state, teacher_state, outs, metrics) = self.opt(modules, self.loss, data, teacher_data, state,teacher_state, traj = traj, teacher_traj = teacher_traj, has_aux=True)
        print("**outs:", outs.keys())
        print("out[post]:", outs["post"])

        metrics.update(mets)
        self.context = {**data, **outs["post"]}
        self.start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), self.context)



        print("self.start:", self.start)

        # mets, (state, outs, metrics) = self.opt(modules, self.imagination_loss, data, state, has_aux=True)

        return state, teacher_state, outs, metrics
    
    def student_imagine_with_actions(self, start, teacher_actions):
        """
        Given an initial student latent 'start' and a batch of teacher_actions,
        rolls out the STUDENT RSSM with exactly those actions for the same horizon.
        Returns a dict with the student latents over that rollout.
        """
        latents = []
        current = start

        # teacher_actions could be shape [horizon, batch_size, action_dim]
        horizon = teacher_actions.shape[0]

        for t in range(horizon):
            current = self.rssm.img_step(current, teacher_actions[t])  
            # current might be a dict like {"deter": ..., "stoch": ...}
            latents.append(current)

        # latents is now a list of dicts (length horizon).
        # We want to stack them into arrays of shape [horizon, batch_size, ...].

        # Example: gather all 'deter' and 'stoch' in separate lists:
        deter_list = [x["deter"] for x in latents]  # each shape [batch_size, deter_dim]
        stoch_list = [x["stoch"] for x in latents]  # each shape [batch_size, stoch_dim, stoch_dim], etc.

        # Now stack them along time:
        rollout_deter = jnp.stack(deter_list, axis=0)   # [horizon, batch_size, deter_dim]
        rollout_stoch = jnp.stack(stoch_list, axis=0)   # [horizon, batch_size, ..., stoch_dim]

        # Construct the dictionary to return
        rollout_dict = {
            "deter": rollout_deter,
            "stoch": rollout_stoch,
            # optionally store the actions, or any other fields you need
            # "actions": teacher_actions,  # if you want them
        }

        return rollout_dict

    def loss(self,traj,teacher_traj, data, teacher_data, state,teacher_state):

        #########################################
        # Change 11/4/2025 19:46 PM
        #########################################

        embed = self.encoder(data)
        teacher_embed = self.teacher_wm.encoder(teacher_data)
        prev_latent, prev_action = state
        teacher_prev_latent, teacher_prev_action = teacher_state

        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)     
        teacher_prev_actions = jnp.concatenate([teacher_prev_action[:, None], teacher_data["action"][:, :-1]], 1)
        teacher_post, teacher_prior = self.teacher_wm.rssm.observe(teacher_embed, teacher_prev_actions, teacher_data["is_first"], teacher_prev_latent)
        # teacher_post  = jax.tree_map(sg, teacher_post)
        # teacher_prior = jax.tree_map(sg, teacher_prior)

        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)

        # embed = self.encoder(data)
        # prev_latent, prev_action = state

        # prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)

        # # Teacher on the *same* sequence
        # teacher_embed = self.teacher_wm.encoder(data)
        # teacher_prev_latent, teacher_prev_action = teacher_state
        # teacher_prev_actions = prev_actions

        # teacher_post, teacher_prior = self.teacher_wm.rssm.observe(
        #     teacher_embed, teacher_prev_actions, data["is_first"], teacher_prev_latent)
        # teacher_post  = jax.tree_map(sg, teacher_post)
        # teacher_prior = jax.tree_map(sg, teacher_prior)

        # # Student
        # post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)

        ##############################################
        # End Change 11/2/2025 2:50 PM
        ##############################################

        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        print("data:", data.keys())
        for key, dist in dists.items():
            # print("key:", key)
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss

        teacher_deter = teacher_post["deter"]   # shape (16, 64, 4096)
        teacher_stoch = teacher_post["stoch"]   # shape (16, 64, 32, 32)

        student_deter = post["deter"]   # shape (16, 64, 4096)
        student_stoch = post["stoch"]   # shape (16, 64, 32, 32)

        total_kl = 0
        N = student_stoch.shape[2]  # e.g. 32
        for i in range(N):  # e.g. 32
            teacher_probs_i = jax.nn.softmax(teacher_stoch[:, :, i, :], axis=-1) 
            student_probs_i = jax.nn.softmax(student_stoch[:, :, i, :], axis=-1)

            teacher_dist_i = distrax.Categorical(probs=teacher_probs_i)
            student_dist_i = distrax.Categorical(probs=student_probs_i)

            kl_i = teacher_dist_i.kl_divergence(student_dist_i)  # shape [T,B]
            total_kl += kl_i  # Sum over factors

        
        #########################################
        # Change 11/4/2025 19:46 PM
        #########################################

        losses["posterior_stoch_kl"] = jnp.mean(total_kl)

        losses["posterior_deter_kl"] = jnp.mean((teacher_deter - student_deter) ** 2)

        teacher_deter = teacher_prior["deter"]   # shape (16, 64, 4096)
        teacher_stoch = teacher_prior["stoch"]   # shape (16, 64, 32, 32)

        student_deter = prior["deter"]   # shape (16, 64, 4096)
        student_stoch = prior["stoch"]   # shape (16, 64, 32, 32)

        losses["prior_deter_kl"] = jnp.mean((teacher_deter - student_deter) ** 2)
        
        total_kl = 0
        N = student_stoch.shape[2]  # e.g. 32
        for i in range(N):  # e.g. 32
            teacher_probs_i = jax.nn.softmax(teacher_stoch[:, :, i, :], axis=-1) 
            student_probs_i = jax.nn.softmax(student_stoch[:, :, i, :], axis=-1)

            teacher_dist_i = distrax.Categorical(probs=teacher_probs_i)
            student_dist_i = distrax.Categorical(probs=student_probs_i)

            kl_i = teacher_dist_i.kl_divergence(student_dist_i)  # shape [T,B]
            total_kl += kl_i  # Sum over factors

        losses["prior_stoch_kl"] = jnp.mean(total_kl)

        # post_t  = self.teacher_wm.rssm.get_dist(teacher_post)
        # prior_t = self.teacher_wm.rssm.get_dist(teacher_prior)
        # post_s  = self.rssm.get_dist(post)
        # prior_s = self.rssm.get_dist(prior)

        # losses["posterior_kl"] = post_t.kl_divergence(post_s).mean()
        # losses["prior_kl"]     = prior_t.kl_divergence(prior_s).mean()


        #########################################
        # End Change 11/2/2025 2:50 PM
        #########################################


        if traj is None or teacher_traj is None:
            pass
        else:
            kl_logit_list = []
            kl_stoch_list = []
            kl_deter_list = []
            for t in range(self.config.imag_horizon):

                teacher_dist = self.teacher_wm.rssm.get_dist({"logit": teacher_traj["logit"][t]})
                student_dist = self.rssm.get_dist({"logit": traj["logit"][t]})
                kl_logit_list.append(teacher_dist.kl_divergence(student_dist))
                kl_stoch_list.append(jnp.mean((teacher_traj["stoch"][t] - traj["stoch"][t]) ** 2))
                kl_deter_list.append(jnp.mean((teacher_traj["deter"][t] - traj["deter"][t]) ** 2))

            kl_logit_arr = jnp.stack(kl_logit_list, axis=0)            # [H, batch]
            losses["dist_loss_imagined"] = kl_logit_arr.mean()   # average across time & batch

            kl_stoch_arr = jnp.stack(kl_stoch_list, axis=0)            # [H, batch]
            losses["dist_stoch_imagined"] = kl_stoch_arr.mean()
        
            kl_deter_arr = jnp.stack(kl_deter_list, axis=0)            # [H, batch]
            losses["dist_deter_imagined"] = kl_deter_arr.mean()


        distill = {}
        if "posterior_stoch_kl" in losses:
            distill["wm/post_kl_stoch"]  = losses["posterior_stoch_kl"]
        if "posterior_deter_kl" in losses:
            distill["wm/post_mse_deter"] = losses["posterior_deter_kl"]
        if "prior_stoch_kl" in losses:
            distill["wm/prior_kl_stoch"] = losses["prior_stoch_kl"]
        if "prior_deter_kl" in losses:
            distill["wm/prior_mse_deter"] = losses["prior_deter_kl"]
        if "dist_loss_imagined" in losses:
            distill["wm/imag_logit_kl"]   = losses["dist_loss_imagined"]
        if "dist_stoch_imagined" in losses:
            distill["wm/imag_stoch_mse"]  = losses["dist_stoch_imagined"]
        if "dist_deter_imagined" in losses:
            distill["wm/imag_deter_mse"]  = losses["dist_deter_imagined"]

        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())

        #########################################
        # Change 11/4/2025 19:46 PM
        #########################################

        # self.context        = {**data, **post}
        # self.start          = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), self.context)


        # self.teacher_context = {**data, **teacher_post}
        # self.teacher_start   = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), self.teacher_context)

        #########################################
        # End Change 11/2/2025 2:50 PM
        #########################################




        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})

        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        teacher_last_latent = {k: v[:, -1] for k, v in teacher_post.items()}
        teacher_last_action = teacher_data["action"][:, -1]
        state = last_latent, last_action
        teacher_state = teacher_last_latent, teacher_last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization
        metrics.update({f"distill/{k}": v for k, v in distill.items()})
        return model_loss.mean(), (state,teacher_state, out, metrics)
    
    def imagination_loss(self, data, state):
        embed = self.encoder(data)
        teacher_embed = self.teacher_wm.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        teacher_post, teacher_prior = self.teacher_wm.rssm.observe(teacher_embed, prev_actions, data["is_first"], prev_latent)
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        
        teacher_latent0 = self.start

        print("teacher_latent0:", teacher_latent0)

        print("teacher_policy:", self.teacher_policy)   

        if self.teacher_policy is None or teacher_latent0 is None:
            pass
        else:
            teacher_traj = self.teacher_wm.imagine(
                self.teacher_policy, teacher_latent0, horizon=self.config.imag_horizon
            )

            print("pass")

            student_latent0 = self.start
            student_traj = self.student_imagine_with_actions(
                student_latent0,
                teacher_traj["action"],  # feed teacher actions
            )

            losses["dist_loss_imagined"] = jnp.mean(
                (teacher_traj["deter"] - student_traj["deter"])**2
            )

            losses["dist_loss_imagined_stoch"] = jnp.mean(
                (teacher_traj["stoch"] - student_traj["stoch"])**2
            )

        # losses['dist_loss'] = jnp.mean((teacher_post['deter'] - post['deter'])**2)
        
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization
        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        print("1. agent imagination start:", start.keys())
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        print("1. keys:", keys)
        start = {k: v for k, v in start.items() if k in keys}
        print("2. agent imagination start:", start.keys())
        start["action"] = policy(start)
        print("3. agent imagination start:", start.keys())
        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        traj = jaxutils_student.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        outs, carry = policy(start, carry)
        start["action"] = outs
        start["carry"] = carry

        def step(prev, _):
            prev = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry}

        traj = jaxutils_student.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data, teacher_data):
        state = self.initial(len(data["is_first"]))
        teacher_state = self.initial(len(teacher_data["is_first"]))
        report = {}
        report.update(self.loss(traj=None,teacher_traj=None,data=data,teacher_data=teacher_data, state=state,teacher_state=teacher_state)[-1][-1])
        context, _ = self.rssm.observe(self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils_student.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils_student.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils_student.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        # metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        # metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils_student.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils_student.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, obs_space, act_space, teacher_wm, teacher_policy, config):
        self.teacher_wm = teacher_wm
        self.teacher_policy = teacher_policy
        self.obs_space = obs_space
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        self.actor = nets_student.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {k: jaxutils_student.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        self.opt = jaxutils_student.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry
    

    #########################################
    # Change 11/4/2025 19:46 PM
    #########################################

    def train(self, imagine, start, context, teacher_wm, teacher_policy):
    # def train(self, imagine, start, teacher_start, context, teacher_wm, teacher_policy):
    #########################################
    # End Change 11/2/2025 2:50 PM
    #########################################
        def loss(tra, teacher_tra, start):
            
            def teach_policy(latent):
                outs, new_state = teacher_policy(latent, None)
                action_array = outs["action"].sample(seed=nj.rng())
                return action_array

            losses = {}
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            carry = None
            action = lambda s: teacher_policy(s).sample(seed=nj.rng())


            # dist, new_carry = teacher_policy(s, carry)  
            # action = dist.sample(seed=nj.rng())
            # teacher_policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            #########################################
            # Change 11/4/2025 19:46 PM
            #########################################
            teacher_traj = teacher_wm.imagine(teach_policy, start, self.config.imag_horizon)
            # teacher_traj = teacher_wm.imagine(teach_policy, teacher_start, self.config.imag_horizon)
            #########################################
            # End Change 11/2/2025 2:50 PM
            #########################################
            print("1.traj:", traj.keys())
            print("2.teacher_traj:", teacher_traj.keys())

            


            # teacher_deters = teacher_traj["deter"]  # shape [H, batch, deter_dim]
            # teacher_stochs = teacher_traj["stoch"]  # shape [H, batch, stoch_dim]
            # student_deters = student_traj["deter"]
            # student_stochs = student_traj["stoch"]

            # for t in range(self.config.imag_horizon):

            #     t_dist = self.teacher_wm.rssm.get_dist(teacher_deters[t], teacher_stochs[t])
            #     s_dist = self.rssm.latent_prior(student_deters[t], student_stochs[t])
            #     kl_list.append(t_dist.kl_divergence(s_dist))


            loss, metrics = self.loss(traj,teacher_traj)
            return loss, (traj, teacher_traj, metrics)
        tra = None
        teacher_tra = None
        mets, (traj, teacher_traj, metrics) = self.opt(self.actor, loss, start, traj = tra, teacher_traj=teacher_tra,has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj,teacher_traj, metrics

    def loss(self,traj,teacher_traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils_student.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils_student.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils_student.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        # if len(self.critics) != 1:
        #  raise NotImplementedError('Must have exactly one critic for TD error calculation.')

        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # Store TD error for PER prioritization

        adv = jnp.stack(advs).sum(0)
        
        student_policy = self.actor(sg(traj))


        teacher_dist = self.actor(sg(teacher_traj))

        teacher_actions = teacher_traj["action"]  # shape [T, ...] or similar

        student_actions = traj["action"]
        # kl_div = teacher_dist.log_prob(teacher_actions) \
        #         - student_policy.log_prob(teacher_actions)
        
        teacher_logp = teacher_dist.log_prob(teacher_actions)
        student_logp = student_policy.log_prob(teacher_actions)
        kl_div      = teacher_logp - student_logp          # P_teacher  ||  P_student


        # shape: [time, batch, ...] or [batch, time, ...] depending on your code
        # you can reduce mean over time/batch:
        kl_div_mean = kl_div.mean()

        policy = student_policy

        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))

        kl_coef = self.config.kl_coef if hasattr(self.config, "kl_coef") else 2.0
        # add the KL to the total loss
        # loss += kl_coef * kl_div[:-1]  # or .mean() if you prefer a scalar

        # For logging
        metrics["kl_mean"] = (kl_coef * kl_div[:-1]).mean()
        metrics["kl_div"] = kl_div
        metrics.update(jaxutils_student.tensorstats(kl_div, "distill/actor/policy_kl"))


        if key == "teacher":
            jax.debug.print("Teacher reward = {}", rew)
            jax.debug.print("Teacher reward mean = {}", rew.mean())
            jax.debug.print("Teacher reward std = {}", rew.std())
            # print("************************************" , rew.mean())
            # print("************************************" , rew.std())

        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils_student.tensorstats(act, "action"))
        metrics.update(jaxutils_student.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils_student.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils_student.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils_student.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils_student.subsample(traj["weight"])
        return metrics

class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets_student.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets_student.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils_student.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils_student.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        tra = None
        teacher_tra = None
        mets, metrics = self.opt(self.net, self.loss, traj, target,traj=tra, teacher_traj=teacher_tra,  has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self,tra, teacher_tra, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils_student.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]

