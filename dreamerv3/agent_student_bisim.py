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
    
# class BisimCalib(nj.Module):
#     def __init__(self, name="bisim_calib"):
#         self.scale = nj.Variable(jnp.ones, (), jnp.float32, name="scale")
#         self.bias  = nj.Variable(jnp.zeros, (), jnp.float32, name="bias")
#         # self.scale = nj.Param(jnp.ones([], jnp.float32), name="scale")
#         # self.bias  = nj.Param(jnp.zeros([], jnp.float32), name="bias")
#     def __call__(self, gap):
#         scale = self.scale.read()
#         bias = self.bias.read()
#         return scale * gap + bias
#         # return self.scale * gap + self.bias

class BisimCalib(nj.Module):
    def __init__(self, name="bisim_calib"):
        self._raw_scale = nj.Variable(jnp.zeros, (), jnp.float32, name="raw_scale")
        self._bias      = nj.Variable(jnp.zeros, (), jnp.float32, name="bias")
    def __call__(self, gap):
        scale = jax.nn.softplus(self._raw_scale.read()) + 1e-4
        bias  = jnp.maximum(self._bias.read(), 0.0)
        return scale * gap + bias

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
        self.encoder = nets_student.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets_student.RSSM(**config.rssm, name="rssm")

        dec_shapes = {k: v for k, v in shapes.items() if k != "desired_goal"}

        self.heads = {
            "decoder": nets_student.MultiDecoder(dec_shapes, **config.decoder, name="dec"),
            "reward": nets_student.MLP((), **config.reward_head, name="rew"),
            "cont": nets_student.MLP((), **config.cont_head, name="cont"),     
        }

        self.opt = jaxutils_student.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

        ####################################################################################################
        # Bisim specific modules and variables
        ####################################################################################################
        
        # self.bisim_scale = nj.Variable(jnp.ones([], jnp.float32),  name="bisim_scale")
        # self.bisim_bias  = nj.Variable(jnp.zeros([], jnp.float32), name="bisim_bias")
        # self.dT_ema      = nj.Variable(jnp.ones([], jnp.float32),  name="dT_ema")

        self.dT_ema = nj.Variable(jnp.ones, (), jnp.float32, name="dT_ema")

        self.phi_head = nets_student.MLP(
            name="phi", dims="deter", shape=(self.config.bisim_dim,),
            **self.config.phi_head
        )

        # self.dT_ema = getattr(self, "dT_ema", 1.0)

        self.bisim_calib = BisimCalib(name="bisim_calib")

        ###################################################################################################

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action
        

    def train(self, data, teacher_data, state,teacher_state,traj=None, teacher_traj=None):
        modules = [self.encoder, self.rssm, self.phi_head, self.bisim_calib, *self.heads.values()]
        mets, (state, teacher_state, outs, metrics) = self.opt(modules, self.loss, data, teacher_data, state,teacher_state, traj = traj, teacher_traj = teacher_traj, has_aux=True)
        metrics.update(mets)
        self.context = {**data, **outs["post"]}
        self.start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), self.context)
        return state, teacher_state, outs, metrics
    
    def student_imagine_with_actions(self, start, teacher_actions):
        latents = []
        current = start
        horizon = teacher_actions.shape[0]

        for t in range(horizon):
            current = self.rssm.img_step(current, teacher_actions[t])  
            latents.append(current)

        deter_list = [x["deter"] for x in latents]  
        stoch_list = [x["stoch"] for x in latents]  

        rollout_deter = jnp.stack(deter_list, axis=0)   
        rollout_stoch = jnp.stack(stoch_list, axis=0)   

        rollout_dict = {
            "deter": rollout_deter,
            "stoch": rollout_stoch,
        }

        return rollout_dict

    def loss(self,traj,teacher_traj, data, teacher_data, state,teacher_state):

        embed = self.encoder(data)
        teacher_embed = self.teacher_wm.encoder(teacher_data)
        prev_latent, prev_action = state
        teacher_prev_latent, teacher_prev_action = teacher_state

        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)     
        teacher_prev_actions = jnp.concatenate([teacher_prev_action[:, None], teacher_data["action"][:, :-1]], 1)
        teacher_post, teacher_prior = self.teacher_wm.rssm.observe(teacher_embed, teacher_prev_actions, teacher_data["is_first"], teacher_prev_latent)
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)

        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
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

            kl_i = teacher_dist_i.kl_divergence(student_dist_i)  
            total_kl += kl_i  

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

        ####################################################################################
        # ---- Bisimulation-style pairwise loss (teacher-targeted, student embedding) ----
        ####################################################################################

        EPS = 1e-6
        gamma_bisim = getattr(self.config, "bisim_gamma", 0.99)
        alpha_r     = getattr(self.config, "bisim_alpha_r", 1.0)
        w_deter     = getattr(self.config, "bisim_w_deter", 1.0)
        w_stoch     = getattr(self.config, "bisim_w_stoch", 1.0)

        # 1) Teacher "next-latent" (one-step prior)
        t_deter = teacher_prior["deter"]                          # [T,B,Dd]
        t_probs = jax.nn.softmax(teacher_prior["stoch"], -1)      # [T,B,G,C]

        # 2) Create O(B) pairs by a permutation (per time step)
        B = t_deter.shape[1]
        perm = jax.random.permutation(nj.rng(), B)

        # 3) Student embedding phi(s) from posterior (current state)
        raw_phi = self.phi_head(feats).mean()               # [T,B,D]
        phi_s   = raw_phi / (jnp.linalg.norm(raw_phi, axis=-1, keepdims=True) + 1e-6)
        phi_i, phi_j = phi_s, phi_s[:, perm]
        phi_gap = jnp.linalg.norm(phi_i - phi_j, axis=-1)


        # 4) Teacher bisimulation target d_T(i,j)
        if "reward" in teacher_data:
            r_T = teacher_data["reward"]
            if r_T.ndim == 3 and r_T.shape[-1] == 1:
                r_T = r_T[..., 0]                                 # [T,B]
            dr = jnp.abs(r_T - r_T[:, perm])                      # [T,B]
        else:
            t_feats = {**teacher_post, "embed": teacher_embed}
            r_pred = self.teacher_wm.heads["reward"](jax.lax.stop_gradient(t_feats)).mean()[..., 0]  # [T,B]
            dr = jnp.abs(r_pred - r_pred[:, perm])

        #########################################################################################
        # ----------------- BEGIN PATCH: multi-action + multi-step bisimulation -----------------
        #########################################################################################

        EPS = 1e-6
        mode      = getattr(self.config, "bisim_mode", "single")        # "single" | "multi"
        K_actions = int(getattr(self.config, "bisim_K_actions", 1))
        agg       = getattr(self.config, "bisim_agg", "softmax")        # "softmax" | "max" | "mean"
        tau       = float(getattr(self.config, "bisim_tau", 0.5))

        H         = int(getattr(self.config, "bisim_ms_horizon", 1))    # multi-step horizon
        ms_gamma  = float(getattr(self.config, "bisim_ms_discount", 0.99))

        # Horizon guard (Python ints so we can use range(H_eff) safely under JIT)
        T      = int(teacher_post["deter"].shape[0])
        H_eff  = max(1, min(H, T))
        T_eff  = T - (H_eff - 1)

        # Helpers
        def js_divergence(p, q, eps=EPS):
            m = 0.5 * (p + q)
            kl_pm = jnp.sum(p * (jnp.log(p + eps) - jnp.log(m + eps)), axis=-1)
            kl_qm = jnp.sum(q * (jnp.log(q + eps) - jnp.log(m + eps)), axis=-1)
            return 0.5 * (kl_pm + kl_qm)  # [...,]
        
        def huber(x, delta=1.0):
            a = jnp.abs(x)
            return jnp.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))

        def _img_step_all(rssm, prev_latent, actions):
            """prev_latent keys [T_eff,B,...], actions [T_eff,B,A] -> next_latent [T_eff,B,...]."""
            TB = prev_latent["deter"].shape[0] * prev_latent["deter"].shape[1]
            prev_flat = jax.tree_map(lambda x: x.reshape((TB,) + x.shape[2:]), prev_latent)
            a_flat    = actions.reshape((TB, -1))
            nxt_flat  = rssm.img_step(prev_flat, a_flat)
            return jax.tree_map(lambda x: x.reshape(prev_latent["deter"].shape[:2] + x.shape[1:]), nxt_flat)

        # One-step baseline via img_step from teacher_post using the executed action at t
        tpost_i, tpost_j = teacher_post, jax.tree_map(lambda x: x[:, perm], teacher_post)
        nxt_i = _img_step_all(self.teacher_wm.rssm, tpost_i, teacher_data["action"][:T])  # [T,B,...]
        nxt_j = _img_step_all(self.teacher_wm.rssm, tpost_j, teacher_data["action"][:T])

        deter_gap_1 = jnp.linalg.norm(nxt_i["deter"] - nxt_j["deter"], axis=-1)             # [T,B]
        stoch_gap_1 = js_divergence(jax.nn.softmax(nxt_i["stoch"], -1),
                                    jax.nn.softmax(nxt_j["stoch"], -1)).mean(axis=-1)       # [T,B]

        # --- Multi-step machinery works on T_eff so we can look ahead H steps ---
        T = t_deter.shape[0]
        T_eff = T if H <= 1 else (T - (H - 1))
        # Slice everything to T_eff when doing multi-step
        teacher_post_eff = jax.tree_map(lambda x: x[:T_eff], teacher_post)
        teacher_post_perm_eff = jax.tree_map(lambda x: x[:T_eff, perm], teacher_post)
        # Teacher action sequence for the next H steps
        # actions_seq[h] = teacher_data["action"][h:h+T_eff]  -> shape [H, T_eff, B, A]
        actions_seq = []
        for h in range(H):
            actions_seq.append(teacher_data["action"][h:h+T_eff].astype(jnp.float32))
        actions_seq = jnp.stack(actions_seq, axis=0)  # [H, T_eff, B, A]

        def _decode(flat_idx, n_steer):
            acc = flat_idx // n_steer
            steer = flat_idx % n_steer
            return acc, steer

        def _encode(acc, steer, n_steer):
            return acc * n_steer + steer

        def _build_actions_K_discrete_near_steer(base_onehot_eff, n_steer, n_acc, S, K):
            """
            base_onehot_eff: [T_eff, B, A]  one-hot teacher actions at t (A = n_steer * n_acc)
            Returns: actions_K: [K, T_eff, B, A] candidates that vary steer by {-S, 0, +S}
                    while keeping acc fixed. K <= (2S + 1).
            """
            num_actions = n_steer * n_acc
            idx = jnp.argmax(base_onehot_eff, axis=-1)  # [T_eff, B]

            def neighbors(i):
                acc, steer = _decode(i, n_steer)
                cand = jnp.array([
                    _encode(acc, jnp.clip(steer - S, 0, n_steer - 1), n_steer),
                    _encode(acc, steer,                                  n_steer),
                    _encode(acc, jnp.clip(steer + S, 0, n_steer - 1), n_steer),
                ], dtype=jnp.int32)  # [3]
                return cand[:K]

            cand_idx = jax.vmap(jax.vmap(neighbors))(idx)     # [T_eff, B, K]
            cand_idx = jnp.transpose(cand_idx, (2, 0, 1))     # [K, T_eff, B]
            actions_K = jax.nn.one_hot(cand_idx, num_actions, dtype=jnp.float32)  # [K,T_eff,B,A]
            return actions_K
        
        def _build_actions_K(base_actions_eff, K):
            """Return actions_K: [K, T_eff, B, A] around base_actions_eff (teacher executed)."""
            if getattr(self.act_space, "discrete", False):
                n_steer = int(getattr(self.config, "bisim_n_steer", 5))
                n_acc   = int(getattr(self.config, "bisim_n_acc",   2))
                S       = int(getattr(self.config, "bisim_steer_stride", 1))  # +/- 1 steer bin
                return _build_actions_K_discrete_near_steer(base_actions_eff, n_steer, n_acc, S, K)
            else:
                # (keep your existing continuous branch unchanged)
                delta = float(getattr(self.config, "bisim_delta", 0.25))
                steer_dim = int(getattr(self.config, "bisim_steer_dim", 0))
                base = base_actions_eff
                a_plus  = base.at[..., steer_dim].add(delta)
                a_minus = base.at[..., steer_dim].add(-delta)
                cand = [base, jnp.clip(a_plus, -1.0, 1.0), jnp.clip(a_minus, -1.0, 1.0)]
                if K > 3:
                    eps = 0.1 * delta
                    cand.append(jnp.clip(base + eps, -1.0, 1.0))
                return jnp.stack(cand[:K], axis=0)  # [K, T_eff, B, A]

        def _rollout_gap_for_actionsK(actionsK_first_step):
            """
            actionsK_first_step: [K,T_eff,B,A/C] candidates for step-0 only.
            Build full H-step sequences by replacing actions_seq[0] with candidate then
            roll H steps with teacher RSSM; compute discounted gap across steps; aggregate over K.
            """
            # For each candidate k, build full action sequence [H,T_eff,B,A]
            def seq_for_k(a0_k):
                return actions_seq.at[0].set(a0_k)  # replace step-0 with candidate

            def gap_for_k(a0_k):
                seq = seq_for_k(a0_k)                  # [H,T_eff,B,A]
                lat_i = teacher_post_eff
                lat_j = teacher_post_perm_eff
                det_gaps = []
                st_gaps = []
                for h in range(H):
                    lat_i = _img_step_all(self.teacher_wm.rssm, lat_i, seq[h])       # next at step h+1
                    lat_j = _img_step_all(self.teacher_wm.rssm, lat_j, seq[h])
                    det_gaps.append(jnp.linalg.norm(lat_i["deter"] - lat_j["deter"], axis=-1))  # [T_eff,B]
                    st_gaps.append(js_divergence(jax.nn.softmax(lat_i["stoch"], -1),
                                                jax.nn.softmax(lat_j["stoch"], -1)).mean(axis=-1))  # [T_eff,B]
                det_gaps = jnp.stack(det_gaps, axis=0)  # [H,T_eff,B]
                st_gaps  = jnp.stack(st_gaps,  axis=0)  # [H,T_eff,B]
                # discounted sum over steps 1..H
                h_w = (ms_gamma ** jnp.arange(1, H + 1)).reshape(H, 1, 1)
                det_sum = (h_w * det_gaps).sum(axis=0)  # [T_eff,B]
                st_sum  = (h_w * st_gaps).sum(axis=0)   # [T_eff,B]
                return w_deter * det_sum + w_stoch * st_sum  # [T_eff,B]

            gaps = jax.vmap(gap_for_k)(actionsK_first_step)  # [K,T_eff,B]
            if agg == "max":
                return jnp.max(gaps, axis=0)                 # [T_eff,B]
            elif agg == "mean":
                return jnp.mean(gaps, axis=0)                # [T_eff,B]
            else:
                return jax.scipy.special.logsumexp(gaps / tau, axis=0) * tau  # [T_eff,B]

        if (mode == "multi") and (K_actions > 1):
            base0_eff = actions_seq[0][:T_eff]                                   # [T_eff,B,A/C]
            actionsK  = _build_actions_K(base0_eff, K_actions)           # [K,T_eff,B,A/C]
            trans_gap_eff = _rollout_gap_for_actionsK(actionsK)          # [T_eff,B]
            # For positions where multi-step cannot be computed (last H-1 steps), fall back to one-step
            # Build a full [T,B] trans_gap by concatenation
            trans_gap = jnp.concatenate([
                trans_gap_eff,
                (w_deter * deter_gap_1 + w_stoch * stoch_gap_1)[T_eff:]
            ], axis=0)  # [T,B]
        else:
            # single-action, one-step gap (your original)
            trans_gap = w_deter * deter_gap_1 + w_stoch * stoch_gap_1    # [T,B]

        # Build the teacher bisimulation target
        dT = alpha_r * dr + gamma_bisim * trans_gap                      # [T,B]

        # Optional: mask terminals/episode starts
        mask = (1.0 - data["is_terminal"].astype(jnp.float32)) * (1.0 - data["is_first"].astype(jnp.float32))
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # [T,B]
        dT  = dT  * mask

        batch_mean = (jnp.sum(dT * mask) / (jnp.sum(mask) + 1e-6))
        # self.dT_ema = 0.99 * self.dT_ema + 0.01 * batch_mean
        new_ema = 0.99 * self.dT_ema.read() + 0.01 * batch_mean
        self.dT_ema.write(new_ema)                # or nj.assign(self.dT_ema, new_ema)
        dT_norm = dT / (self.dT_ema.read() + 1e-6)

        # scale = self.bisim_scale.value
        # bias  = self.bisim_bias.value
        # pred  = scale * phi_gap + bias

        pred = self.bisim_calib(phi_gap)
        err   = pred - jax.lax.stop_gradient(dT_norm)
        losses["bisim_pair"] = (huber(err, 1.0) * mask).mean()

        bisim_pair_loss = huber(phi_gap - jax.lax.stop_gradient(dT_norm), delta=1.0) * mask
                
        ###############################################################################################################

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

        # def _pearson(x, y, eps=1e-8):
        #     x = (x - x.mean()) / (x.std() + eps)
        #     y = (y - y.mean()) / (y.std() + eps)
        #     return (x * y).mean()
        
        # valid = mask > 0.5
        # x = phi_gap[valid]
        # y = dT[valid]

        # valid = (mask > 0.5).astype(phi_gap.dtype)
        # x = phi_gap * valid + (1 - valid) * 0.0  # or use jnp.where(valid, phi_gap, 0)
        # y = dT * valid

        w = mask.astype(phi_gap.dtype)
        wsum = jnp.sum(w) + 1e-8

        x_mean = jnp.sum(w * phi_gap) / wsum
        y_mean = jnp.sum(w * dT) / wsum
        x_std  = jnp.sqrt(jnp.sum(w * (phi_gap - x_mean)**2) / wsum + 1e-8)
        y_std  = jnp.sqrt(jnp.sum(w * (dT      - y_mean)**2) / wsum + 1e-8)
        cov_xy = jnp.sum(w * (phi_gap - x_mean) * (dT - y_mean)) / wsum
        metrics["bisim_pair_corr"] = cov_xy / (x_std * y_std + 1e-8)

        metrics["bisim_phi_gap_mean"] = x_mean
        metrics["bisim_phi_gap_std"]  = x_std
        metrics["bisim_target_mean"]  = y_mean
        metrics["bisim_target_std"]   = y_std

        # def _pearson_masked(x, y, eps=1e-8):
        #     x = (x - x.mean()) / (x.std() + eps)
        #     y = (y - y.mean()) / (y.std() + eps)
        #     return (x * y).mean()
        
        # metrics["bisim_pair_corr"] = _pearson_masked(x, y)

        # def _spearman_masked(x, y, eps=1e-8):
        #     rx = jnp.argsort(jnp.argsort(x))
        #     ry = jnp.argsort(jnp.argsort(y))
        #     rx = (rx - rx.mean()) / (rx.std() + eps)
        #     ry = (ry - ry.mean()) / (ry.std() + eps)
        #     return (rx * ry).mean()
        
        # metrics["bisim_pair_spearman"] = _spearman_masked(x, y)

        dtype = phi_gap.dtype
        metrics["bisim_mode"] = jnp.asarray(1.0 if mode == "multi" else 0.0, dtype)
        metrics["bisim_K"]    = jnp.asarray(float(K_actions), dtype)
        metrics["bisim_ms_H"] = jnp.asarray(float(H), dtype)

        # metrics["bisim_pair_corr"] = _pearson(phi_gap.reshape(-1), dT.reshape(-1))
        # metrics["bisim_mode"]      = jnp.array(1.0 if mode == "multi" else 0.0)
        # metrics["bisim_K"]         = jnp.array(float(K_actions))
        # metrics["bisim_ms_H"]      = jnp.array(float(H))

        metrics["bisim_phi_gap_mean"] = x.mean()
        metrics["bisim_phi_gap_std"]  = x.std()
        metrics["bisim_target_mean"]  = y.mean()
        metrics["bisim_target_std"]   = y.std()
        # metrics["bisim_scale"] = self.bisim_scale.value
        # metrics["bisim_bias"]  = self.bisim_bias.value
        # metrics["bisim_scale"] = jnp.asarray(self.bisim_scale, jnp.float32)
        # metrics["bisim_bias"]  = jnp.asarray(self.bisim_bias,  jnp.float32)
        # metrics["bisim_scale"] = jnp.asarray(self.bisim_calib.scale, jnp.float32)
        # metrics["bisim_bias"]  = jnp.asarray(self.bisim_calib.bias,  jnp.float32)

        metrics["bisim_scale"] = self.bisim_calib.scale.read()
        metrics["bisim_bias"] = self.bisim_calib.bias.read()


        metrics.update(jaxutils_student.tensorstats(bisim_pair_loss, "bisim_pair"))  


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

        # print("teacher_latent0:", teacher_latent0)

        # print("teacher_policy:", self.teacher_policy)   

        if self.teacher_policy is None or teacher_latent0 is None:
            pass
        else:
            teacher_traj = self.teacher_wm.imagine(
                self.teacher_policy, teacher_latent0, horizon=self.config.imag_horizon
            )

            # print("pass")

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
        # print("1. agent imagination start:", start.keys())
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        # print("1. keys:", keys)
        start = {k: v for k, v in start.items() if k in keys}
        # print("2. agent imagination start:", start.keys())
        start["action"] = policy(start)
        # print("3. agent imagination start:", start.keys())
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
            # print("1.traj:", traj.keys())
            # print("2.teacher_traj:", teacher_traj.keys())

            


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


        # if key == "teacher":
            # jax.debug.print("Teacher reward = {}", rew)
            # jax.debug.print("Teacher reward mean = {}", rew.mean())
            # jax.debug.print("Teacher reward std = {}", rew.std())
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

