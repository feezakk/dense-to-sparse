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

from . import behaviors_student_bisim, jaxagent_student_bisim, jaxutils_student, nets_student
from . import ninjax as nj

@jaxagent_student_bisim.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, teacher_wm, teacher_policy, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.teacher_policy = teacher_policy
        self.step = step
        self.start = None
        self.context = None

         # -------------------------------------------------------------
        # 1) FROZEN TEACHER WORLD MODEL
        #    - Separate module name: "teacher_wm"
        #    - Parameters will live under: "agent/teacher_wm/...".
        #    - We will copy pretrained teacher.ckpt weights into this.
        #    - It is NEVER given to any optimizer, so it stays frozen.
        # -------------------------------------------------------------
        self.teacher_wm = WorldModel(
            obs_space,
            act_space,
            teacher_wm=None,        # teacher-of-teacher is None
            teacher_policy=None,    # we don't use a teacher for this module
            start=None,
            context=None,
            config=config,
            name="teacher_wm",      # <--- distinct namespace
        )

        # -------------------------------------------------------------
        # 2) TRAINABLE STUDENT WORLD MODEL
        #    - Uses self.teacher_wm for distillation/bisimulation.
        #    - Parameters live under: "agent/wm/...".
        # -------------------------------------------------------------
        self.wm = WorldModel(
            obs_space, 
            act_space, 
            self.teacher_wm, 
            teacher_policy, 
            self.start, 
            self.context, 
            config, 
            name="wm"
        )

        self.task_behavior = behaviors_student_bisim.Greedy(self.wm, self.obs_space, self.act_space, self.teacher_wm, teacher_policy, self.config, name="task_behavior")
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
        print_teacher_params_norm(self.teacher_wm, label="Before training")

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
        # self.expl_behavior.policy(latent, expl_state)
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
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        teacher_data = self.preprocess(teacher_data)

        # ------------------------------------------------------------------
        # Phase 1: standard WM update from real data only
        # No traj/teacher_traj passed here => no bisim losses in this phase.
        # ------------------------------------------------------------------

        state, teacher_state, wm_outs, mets = self.wm.train(data, teacher_data, state,teacher_state,traj=None, teacher_traj=None)
        metrics.update(mets)

        # ------------------------------------------------------------------
        # RL / actor‑critic training, generates traj, teacher_traj
        # ------------------------------------------------------------------

        context = {**data, **teacher_data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1]+list(x.shape[2:])), {**data, **wm_outs["post"]})
        start_t = tree_map(lambda x: x.reshape([-1]+list(x.shape[2:])), {**teacher_data, **wm_outs["teacher_post"]})

        traj, teacher_traj, mets_expl = self.task_behavior.train(self.wm.imagine, start, start_t, context)

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

        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        # ------------------------------------------------------------------
        # Phase 2: PCB‑only WM update (bisim + distill terms only)
        # ------------------------------------------------------------------

        state2, teacher_state2, outs2, mets2 = self.wm.train(data, teacher_data, state, 
                                                                teacher_state, traj=traj, teacher_traj=sg(teacher_traj), pcb_only=True)
        
        metrics.update({f"pcb_{k}": v for k, v in mets2.items()})

        return traj, teacher_traj, outs2, state2, teacher_state2, metrics

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
        return obs
    
class BisimCalib(nj.Module):
    def __init__(self, name="bisim_calib"):
        self.scale = nj.Variable(jnp.ones, (), jnp.float32, name="scale")
        self.bias = nj.Variable(jnp.zeros, (), jnp.float32, name="bias")

    def ensure_initialized(self):
        if nj.creating():
            self.scale.read()
            self.bias.read()

    def __call__(self, gap):
        self.ensure_initialized()
        scale = jax.nn.softplus(self.scale.read()) + 1e-6
        bias  = jnp.maximum(self.bias.read(), 0.0)
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
        
        self.dT_ema = nj.Variable(jnp.ones, (), jnp.float32, name="dT_ema")

        self.phi_head = nets_student.MLP(
            name="phi", dims="deter", shape=(self.config.bisim_dim,),
            **self.config.phi_head
        )

        self.bisim_calib = BisimCalib(name="bisim_calib")

        ###################################################################################################

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))

        # Ensure φ-head and bisim_calib parameters exist during the creation pass.
        # This runs when JAXAgent._init_varibs calls train_initial -> wm.initial.
        # if nj.creating():
        #     # Create dT_ema in the state.
        #     # _ = self.dT_ema.read()
        #     # φ-head: it expects a dict with "deter" (same as in loss)
        #     dummy_post = {"deter": prev_latent["deter"]}
        #     _ = self.phi_head(dummy_post).mean()

        #     # BisimCalib: only cares about shape; [T,B] vs [1,B] is fine
        #     dummy_gap = jnp.zeros((1, batch_size), dtype=jnp.float32)
        #     _ = self.bisim_calib(dummy_gap)

        return prev_latent, prev_action
        

    def train(self, data, teacher_data, state,teacher_state,traj=None, teacher_traj=None,pcb_only=False):
        modules = [self.encoder, self.rssm, self.phi_head, self.bisim_calib, *self.heads.values()]

        # # --- Creation pass: called from _init_varibs (create=True) ---
        # # Here the NinJAX state is still being constructed.
        # # We must ensure that each module in `modules` has at least one
        # # state entry *before* nj.grad tries to query it via mod.getm().
        # if nj.creating():
        #     # Forward-only pass to instantiate encoder/rssm/heads, etc.
        #     # We ignore the outputs; we only care about side effects on state.
        #     _ = self.loss(
        #         traj,
        #         teacher_traj,
        #         data,
        #         teacher_data,
        #         state,
        #         teacher_state,
        #         pcb_only=pcb_only,
        #     )
        
        mets, (state, teacher_state, outs, metrics) = self.opt(
            modules, self.loss, 
            data, teacher_data, state,teacher_state, 
            traj = traj, 
            teacher_traj = teacher_traj, 
            pcb_only=pcb_only,
            has_aux=True)
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
    
    def imagine_with_actions(self, start, actions):
        """Roll RSSM with a fixed action sequence.
        start: dict of latent tensors (batch B)
        actions: [T, B, A]
        returns: traj dict with latents [T1, B, ...] and 'action' [T, B, A]
        """
        def step(prev, a):
            return self.rssm.img_step(prev, a), None
        T, B = actions.shape[:2]
        lat0 = {k: start[k] for k in self.rssm.initial(1).keys()}
        def scan_step(carry, a):
            nxt = self.rssm.img_step(carry, a)
            return nxt, nxt
        state = lat0
        ys = []
        for t in range(T):
            state, _ = scan_step(state, actions[t])
            ys.append(state)
        traj = {k: jnp.concatenate([lat0[k][None], jnp.stack([y[k] for y in ys], 0)], 0)
                for k in lat0.keys()}
        traj["action"] = actions
        return traj

    def loss(self,traj,teacher_traj, data, teacher_data, state,teacher_state,pcb_only=False):

        embed = self.encoder(data)
        teacher_embed = self.teacher_wm.encoder(teacher_data)
        prev_latent, prev_action = state
        teacher_prev_latent, teacher_prev_action = teacher_state

        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)     
        teacher_prev_actions = jnp.concatenate([teacher_prev_action[:, None], teacher_data["action"][:, :-1]], 1)
        teacher_post, teacher_prior = self.teacher_wm.rssm.observe(teacher_embed, teacher_prev_actions, teacher_data["is_first"], teacher_prev_latent)
        teacher_post  = jax.tree_map(sg, teacher_post)
        teacher_prior = jax.tree_map(sg, teacher_prior)
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)

        # --- Ensure bisim-specific modules have parameters on first create pass ---
        if nj.creating():
            # 1) scalar EMA
            _ = self.dT_ema.read()

            # 2) φ-head
            # Use post['deter'] so shapes are realistic.
            dummy_feat = {"deter": post["deter"]}
            _ = self.phi_head(dummy_feat).mean()

            # 3) calibration head
            dummy_gap = jnp.zeros(post["deter"].shape[:2], dtype=jnp.float32)  # [T, B]
            _ = self.bisim_calib(dummy_gap)
        # -------------------------------------------------------------------------

        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}

        # # ------------------------------------------------------------------
        # # Ensure phi_head and bisim_calib have created parameters even in
        # # the data-only pass (traj=None, teacher_traj=None).
        # # This prevents ninjax.getm() from seeing an “empty” module.
        # # ------------------------------------------------------------------
        # # 1) Call phi_head once on real latents (uses 'deter' inside).
        # _ = self.phi_head(post).mean()

        # # 2) Call bisim_calib on a dummy gap tensor with [T,B] shape.
        # dummy_gap = jnp.zeros(post["deter"].shape[:2], dtype=jnp.float32)
        # _ = self.bisim_calib(dummy_gap)
        # # ------------------------------------------------------------------

        # # dists = {}
        # # feats = {**post, "embed": embed}


        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            target = (teacher_data[key] if key == "reward"
                      else data[key])
            loss = -dist.log_prob(target.astype(jnp.float32))
            # loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss

        teacher_deter = teacher_post["deter"]   # shape (16, 64, 4096)
        teacher_stoch = teacher_post["stoch"]   # shape (16, 64, 32, 32)

        student_deter = post["deter"]   # shape (16, 64, 4096)
        student_stoch = post["stoch"]   # shape (16, 64, 32, 32)

        # Posterior (teacher_post vs post)
        t_probs_post = self.teacher_wm.rssm.get_dist({"logit": teacher_post["logit"]})  # [T,B,G,C]
        s_probs_post = self.rssm.get_dist({"logit": post["logit"]})                     # [T,B,G,C]
        posterior_kl_per_group = t_probs_post.kl_divergence(s_probs_post)

        losses["posterior_stoch_kl"] = posterior_kl_per_group.mean()
        losses["posterior_deter_kl"] = jnp.mean((teacher_post["deter"] - post["deter"]) ** 2)

        # Prior (teacher_prior vs prior)
        t_probs_prior = self.teacher_wm.rssm.get_dist({"logit": teacher_prior["logit"]})
        s_probs_prior = self.rssm.get_dist({"logit": prior["logit"]})
        prior_kl_per_group = t_probs_prior.kl_divergence(s_probs_prior)
        losses["prior_stoch_kl"] = prior_kl_per_group.mean()
        losses["prior_deter_kl"] = jnp.mean((teacher_prior["deter"] - prior["deter"]) ** 2)

        if (traj is not None) and (teacher_traj is not None):
            # Reward match under student actions
            s_rew = self.heads["reward"](traj).mean()           # [T+1,B] or [T,B]
            t_rew = sg(self.teacher_wm.heads["reward"](teacher_traj).mean())
            # Align time dims to [T,B] (exclude last latent if needed)
            if s_rew.shape[0] == teacher_traj["action"].shape[0] + 1:
                s_rew = s_rew[:-1]
            if t_rew.shape[0] == teacher_traj["action"].shape[0] + 1:
                t_rew = t_rew[:-1]
            losses["bisim_imag_reward_mse"] = ((s_rew - t_rew) ** 2).mean()

            # Next-latent distribution match (1..T)
            s_next = {k: v[1:] for k, v in traj.items() if k in ("stoch","logit","deter")}
            t_next = {k: sg(v[1:]) for k, v in teacher_traj.items() if k in ("stoch","logit","deter")}
            s_next_dist = self.rssm.get_dist(s_next)
            t_next_dist = self.teacher_wm.rssm.get_dist(t_next)
            # losses["bisim_imag_next_kl"] = distrax.kl_divergence(s_next_dist, t_next_dist).mean()
            losses["bisim_imag_next_kl"] = s_next_dist.kl_divergence(t_next_dist).mean()
            if "deter" in s_next and "deter" in t_next:
                losses["bisim_imag_next_deter"] = jnp.mean((s_next["deter"] - t_next["deter"]) ** 2)

        if pcb_only:
            def _keep_pcb(name: str) -> bool:
                # bisim_* terms (pairwise + imagined alignment)
                if name.startswith("bisim_"):
                    return True
                # teacher distillation terms (posterior/prior KLs, imagined distill)
                # if name.startswith("posterior_"):
                #     return True
                # if name.startswith("prior_"):
                #     return True
                if name.startswith("dist_"):
                    return True
                return False

            losses = {k: v for k, v in losses.items() if _keep_pcb(k)}

        

        out = {"embed": embed, "post": post, "prior": prior, "teacher_embed": teacher_embed, "teacher_post": teacher_post, "teacher_prior": teacher_prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})

        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        teacher_last_latent = {k: v[:, -1] for k, v in teacher_post.items()}
        teacher_last_action = teacher_data["action"][:, -1]
        state = last_latent, last_action
        teacher_state = teacher_last_latent, teacher_last_action

        if (traj is not None) and (teacher_traj is not None):
            # ------------------------------------------------------------------
            # NEW: Rebuild a differentiable student rollout under the SAME action
            # sequence used to produce 'teacher_traj'. This ensures gradients
            # reach the student RSSM parameters within this loss call.
            # ------------------------------------------------------------------
            # 1) Starting latent is the first latent in the provided student traj.
            #    It is treated as a constant; gradients flow through transitions.
            rssm_keys = list(self.rssm.initial(1).keys())  # typically ["deter","stoch","logit"]
            # s0 = {k: traj[k][0] for k in rssm_keys if k in traj}  # [B, ...]
            s0 = jax.tree_map(jax.lax.stop_gradient, {k: traj[k][0] for k in rssm_keys if k in traj})
            # 2) Do not propagate gradients to the ACTOR through actions here.
            a_seq = jax.lax.stop_gradient(traj["action"][:-1])    # [T,B,A]
            s_recon = self.imagine_with_actions(s0, a_seq)        # latents [T+1,B,...], "action":[T,B,A]

            # ------------------------------------------------------------------
            # (A) Reward match under student actions (teacher reward head frozen)
            # ------------------------------------------------------------------
            s_rew = self.heads["reward"](s_recon).mean()                # [T+1,B] or [T,B]
            t_rew = sg(self.teacher_wm.heads["reward"](teacher_traj).mean())

            # Align to [T,B] (action length) if a +1 latent exists.
            if s_rew.shape[0] == teacher_traj["action"].shape[0] + 1:
                s_rew = s_rew[:-1]
            if t_rew.shape[0] == teacher_traj["action"].shape[0] + 1:
                t_rew = t_rew[:-1]

            losses["bisim_imag_reward_mse"] = ((s_rew - t_rew) ** 2).mean()

            # ------------------------------------------------------------------
            # (B) Next-latent distribution alignment (student vs teacher)
            # ------------------------------------------------------------------
            s_next = {k: v[1:] for k, v in s_recon.items() if k in ("stoch","logit","deter")}
            t_next = {k: sg(v[1:]) for k, v in teacher_traj.items() if k in ("stoch","logit","deter")}
            s_next_dist = self.rssm.get_dist(s_next)
            t_next_dist = self.teacher_wm.rssm.get_dist(t_next)
            # losses["bisim_imag_next_kl"] = s_next_dist.kl_divergence(t_next_dist).mean()
            losses["bisim_imag_next_kl"] = t_next_dist.kl_divergence(s_next_dist).mean()

            if "deter" in s_next and "deter" in t_next:
                losses["bisim_imag_next_deter"] = jnp.mean((s_next["deter"] - t_next["deter"]) ** 2)

            # ------------------------------------------------------------------
            # (C) Pairwise φ-based bisimulation under STUDENT actions
            # ------------------------------------------------------------------
            def _phi_norm(feat):
                z = self.phi_head(feat).mean()
                return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)

            # Transition gap on TEACHER NEXT latents (stop-grad teacher)
            def _phi_norm_T_no_grad(featT):
                z = self.phi_head(featT).mean()
                z = jax.lax.stop_gradient(z)
                return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)
            
            def huber(x, delta=1.0):
                a = jnp.abs(x)
                return jnp.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))
            
            def _shift_back(x, k):
                """Return time-shifted x[t] = orig[t+k] for t < T-k, else 0."""
                pad = jnp.zeros((k,) + x.shape[1:], dtype=x.dtype)
                return jnp.concatenate([x[k:], pad], axis=0)
            
            K = int(getattr(self.config, "bisim_ms_steps", 4))
            alpha_r = float(getattr(self.config, "bisim_alpha_r", 1.0))
            gamma_bisim = float(getattr(self.config, "bisim_gamma", 0.99))
            gamma_ms = float(getattr(self.config, "bisim_ms_gamma", gamma_bisim))
            clip = float(getattr(self.config, "bisim_target_clip", 10.0))

            # Use CURRENT student latents at time t (exclude the last latent)
            s_now = {k: v[:-1] for k, v in s_recon.items() if k in ("deter","stoch","logit")}  # [T,B,...]
            B = s_now["deter"].shape[1]
            perm = jax.random.permutation(nj.rng(), B)

            s_i = s_now
            s_j = jax.tree_map(lambda x: x[:, perm], s_now)

            phi_now_gap = jnp.linalg.norm(_phi_norm(s_i) - _phi_norm(s_j), axis=-1)  # [T,B]
            # Teacher reward difference at t (teacher under SAME actions)
            t_rew_full = self.teacher_wm.heads["reward"](jax.tree_map(sg, teacher_traj)).mean()
            if t_rew_full.ndim == 3 and t_rew_full.shape[-1] == 1:
                t_rew_full = t_rew_full[..., 0]
            if t_rew_full.shape[0] == teacher_traj["action"].shape[0] + 1:
                t_rew_full = t_rew_full[:-1]
                
            # Mask with teacher continuation (as weight)
            t_cont = self.teacher_wm.heads["cont"](jax.tree_map(sg, teacher_traj)).mode()
            if t_cont.ndim == 3 and t_cont.shape[-1] == 1:
                t_cont = t_cont[..., 0]
            if t_cont.shape[0] == teacher_traj["action"].shape[0] + 1:
                t_cont = t_cont[:-1]

            
        
            t_next_i = jax.tree_map(lambda x: sg(x)[1:], teacher_traj)       # [T,B,...]
            t_next_j = jax.tree_map(lambda x: sg(x)[1:, perm], teacher_traj) # [T,B,...]                                     # [T,B,Dφ]

            # φ(teacher next) at every step (t+1), used for the transition gap
            phi_T_next = _phi_norm_T_no_grad(jax.tree_map(lambda x: sg(x)[1:], teacher_traj))  # [T,B,Dφ]
            phi_T_next_perm = phi_T_next[:, perm]  

            if K <= 1:
                dR = jnp.abs(t_rew_full - jnp.take(t_rew_full, perm, axis=1))  # [T,B]
                trans_gap = jnp.linalg.norm(_phi_norm_T_no_grad(t_next_i) - _phi_norm_T_no_grad(t_next_j), axis=-1)  # [T,B]
                dT = alpha_r * dR + gamma_bisim * trans_gap  # [T,B]
                mask = t_cont.astype(jnp.float32)  # [T,B]

            else:
                # ---- multi-step window target ----
                Ks = jnp.arange(K)
                k_weights = (gamma_ms ** Ks).astype(jnp.float32)  # [K]

                # Build [K,T,B] tensors for reward diffs, transition gaps, and prefix masks
                dR_stack = []
                trans_stack = []
                cont_stack = []
                t_rew_perm = jnp.take(t_rew_full, perm, axis=1)  # [T,B]

                for k in range(K):
                    # reward at t+k for (i, j)
                    dR_k = jnp.abs(_shift_back(t_rew_full, k) - _shift_back(t_rew_perm, k))  # [T,B]
                    # transition gap at t+k (uses teacher next-latents)
                    phi_k     = _shift_back(phi_T_next, k)        # [T,B,Dφ]
                    phi_k_p   = _shift_back(phi_T_next_perm, k)
                    trans_k   = jnp.linalg.norm(phi_k - phi_k_p, axis=-1)                    # [T,B]
                    # prefix continuation mask ∏_{m=0..k} cont[t+m]
                    cont_k    = _shift_back(t_cont, k)                                       # [T,B]

                    dR_stack.append(dR_k)
                    trans_stack.append(trans_k)
                    cont_stack.append(cont_k)

                dR_all      = jnp.stack(dR_stack, 0)          # [K,T,B]
                trans_all   = jnp.stack(trans_stack, 0)       # [K,T,B]
                cont_shifts = jnp.stack(cont_stack, 0)        # [K,T,B]
                mask_all    = jnp.cumprod(cont_shifts, axis=0)  # inclusive product over k

                terms = alpha_r * dR_all + gamma_bisim * trans_all               # [K,T,B]
                dT = jnp.sum((k_weights[:, None, None] * terms) * mask_all, 0)   # [T,B]
                mask = cont_shifts[0]                                           # [T,B]


            # EMA normalization + Huber on calibrated φ-gap
            batch_mean = (jnp.sum(dT * mask) / (jnp.sum(mask) + 1e-6))
            self.dT_ema.write(0.99 * self.dT_ema.read() + 0.01 * batch_mean)
            dT_norm = jnp.clip(dT / (self.dT_ema.read() + 1e-6), 0.0, clip)

            pred = self.bisim_calib(phi_now_gap)  # scale * φ_gap + bias


            err = pred - jax.lax.stop_gradient(dT_norm)
            losses["bisim_pair"] = (huber(err, 1.0) * mask).mean()

            # Metrics
            wsum = jnp.sum(mask) + 1e-8
            x_mean = jnp.sum(mask * phi_now_gap) / wsum
            y_mean = jnp.sum(mask * dT) / wsum
            x_std  = jnp.sqrt(jnp.sum(mask * (phi_now_gap - x_mean)**2) / wsum + 1e-8)
            y_std  = jnp.sqrt(jnp.sum(mask * (dT        - y_mean)**2) / wsum + 1e-8)
            cov_xy = jnp.sum(mask * (phi_now_gap - x_mean) * (dT - y_mean)) / wsum

        #################################################################################################################

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
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization

        if (traj is not None) and (teacher_traj is not None):
            metrics["bisim_pair_corr"]   = cov_xy / (x_std * y_std + 1e-8)
            metrics["bisim_phi_gap_mean"] = x_mean
            metrics["bisim_phi_gap_std"]  = x_std
            metrics["bisim_target_mean"]  = y_mean
            metrics["bisim_target_std"]   = y_std

            scale_eff = jax.nn.softplus(self.bisim_calib.scale.read()) + 1e-6
            bias_eff  = jnp.maximum(self.bisim_calib.bias.read(), 0.0)
            metrics["bisim_scale"] = scale_eff
            metrics["bisim_bias"]  = bias_eff


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
        report.update(self.loss(traj=None,teacher_traj=None,data=data,teacher_data=teacher_data, state=state,teacher_state=teacher_state, pcb_only=False)[-1][-1])
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
    
    def train(self, imagine, start, start_t, context, teacher_wm, teacher_policy):
        def loss(tra, teacher_tra, start, start_t):
            
            # def teach_policy(latent):
            #     outs, new_state = teacher_policy(latent, None)
            #     action_array = outs["action"].sample(seed=nj.rng())
            #     return action_array

            losses = {}
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            # carry = None
            # action = lambda s: teacher_policy(s).sample(seed=nj.rng())

            traj = imagine(policy, start, self.config.imag_horizon)
            # teacher_traj = teacher_wm.imagine(teach_policy, start_t, self.config.imag_horizon)
            # teacher_traj = teacher_wm.imagine(teach_policy, start, self.config.imag_horizon)

            a_seq = sg(traj["action"][:-1])                         # [T, B, A] detach


            def imagine_with_actions(start, actions):
                """Roll RSSM with a fixed action sequence.
                start: dict of latent tensors (batch B)
                actions: [T, B, A]
                returns: traj dict with latents [T1, B, ...] and 'action' [T, B, A]
                """
                def step(prev, a):
                    return teacher_wm.rssm.img_step(prev, a), None
                T, B = actions.shape[:2]
                lat0 = {k: start[k] for k in teacher_wm.rssm.initial(1).keys()}
                def scan_step(carry, a):
                    nxt = teacher_wm.rssm.img_step(carry, a)
                    return nxt, nxt
                state = lat0
                ys = []
                for t in range(T):
                    state, _ = scan_step(state, actions[t])
                    ys.append(state)
                traj = {k: jnp.concatenate([lat0[k][None], jnp.stack([y[k] for y in ys], 0)], 0)
                        for k in lat0.keys()}
                traj["action"] = actions
                return traj


            teacher_traj = imagine_with_actions(start_t, a_seq)
 

            loss, metrics = self.loss(traj,teacher_traj)
            return loss, (traj, teacher_traj, metrics)
        tra = None
        teacher_tra = None
        mets, (traj, teacher_traj, metrics) = self.opt(self.actor, loss, start, start_t, traj = tra, teacher_traj=teacher_tra,has_aux=True)
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


        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # Store TD error for PER prioritization

        adv = jnp.stack(advs).sum(0)
        
        student_policy = self.actor(sg(traj))


        # teacher_dist = self.actor(sg(teacher_traj))

        # teacher_actions = teacher_traj["action"]  # shape [T, ...] or similar

        # student_actions = traj["action"]
        
        # teacher_logp = teacher_dist.log_prob(teacher_actions)
        # student_logp = student_policy.log_prob(teacher_actions)
        # kl_div      = teacher_logp - student_logp          # P_teacher  ||  P_student

        # kl_div_mean = kl_div.mean()

        policy = student_policy

        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))

        kl_coef = self.config.kl_coef if hasattr(self.config, "kl_coef") else 2.0

        # For logging
        # metrics["kl_mean"] = (kl_coef * kl_div[:-1]).mean()
        # metrics["kl_div"] = kl_div
        # metrics.update(jaxutils_student.tensorstats(kl_div, "distill/actor/policy_kl"))

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

