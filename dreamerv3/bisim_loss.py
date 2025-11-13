import jax.numpy as jnp

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

        def _phi_norm(feat):
            z = self.phi_head(feat).mean()
            return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)  # [T,B,Dφ]
        
        def _phi_norm_T_no_grad(featT):
            z = self.phi_head(featT).mean()
            z = jax.lax.stop_gradient(z)
            return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)

        def _rollout_phi_gap_piT_teacher(t_i0, t_j0, H_, key):
            # closed-loop teacher rollout; returns [T_eff,B]
            t_i = jax.tree_map(lambda x: x[:T_eff], t_i0)
            t_j = jax.tree_map(lambda x: x[:T_eff], t_j0)
            acc = jnp.zeros(t_i["deter"].shape[:2], dtype=t_i["deter"].dtype)
            disc = 1.0
            k = key
            for _ in range(H_):
                k, k_i, k_j = jax.random.split(k, 3)
                a_i = _sample_teacher_actions_TB(t_i, k_i)
                a_j = _sample_teacher_actions_TB(t_j, k_j)
                t_i = _img_step_all(self.teacher_wm.rssm, t_i, a_i)
                t_j = _img_step_all(self.teacher_wm.rssm, t_j, a_j)
                disc = disc * ms_gamma
                acc += disc * jnp.linalg.norm(_phi_norm_T_no_grad(t_i) - _phi_norm_T_no_grad(t_j), axis=-1)
            return acc  # [T_eff,B]

        def _teacher_truncated_value(t_i0, H_):
            # dense |ΔVᵀ| proxy via truncated returns from teacher WM & π_T; returns [T_eff,B]
            t = jax.tree_map(lambda x: x[:T_eff], t_i0)
            V = jnp.zeros(t["deter"].shape[:2], dtype=t["deter"].dtype)
            disc = jnp.ones_like(V)
            k = nj.rng()
            for _ in range(H_):
                r = self.teacher_wm.heads["reward"](jax.lax.stop_gradient(t)).mean()
                if r.ndim == 3 and r.shape[-1] == 1:
                    r = r[..., 0]
                V = V + disc * r
                c = self.teacher_wm.heads["cont"](jax.lax.stop_gradient(t)).mode()  # [T_eff,B]
                k, k_a = jax.random.split(k)
                a = _sample_teacher_actions_TB(t, k_a)
                t = _img_step_all(self.teacher_wm.rssm, t, a)
                disc = disc * (gamma_bisim * c)
            return V  # [T_eff,B]


        ####################################################################################
        # ---- Bisimulation-style pairwise loss (teacher-targeted, student embedding) ----
        ####################################################################################

        EPS = 1e-6
        gamma_bisim = getattr(self.config, "bisim_gamma", 0.99)
        alpha_r     = getattr(self.config, "bisim_alpha_r", 1.0)
        w_deter     = getattr(self.config, "bisim_w_deter", 1.0)
        w_stoch     = getattr(self.config, "bisim_w_stoch", 1.0)

        # --- helper: sample teacher actions for a [T,B,...] latent block ---
        def _sample_teacher_actions_TB(latents_TB, key):
            """
            latents_TB: dict with keys 'deter', 'stoch' shaped [T, B, ...]
            Returns: actions [T, B, A] sampled from π_T(·|latents_TB)
            """
            T_, B_ = latents_TB["deter"].shape[:2]
            flat = jax.tree_map(lambda x: x.reshape((T_ * B_,) + x.shape[2:]), latents_TB)
            # stop grads to the teacher policy and teacher WM
            flat = sg(flat)
            outs, _ = self.teacher_policy(flat, None)            # outs["action"] is a distrax dist
            a_flat = outs["action"].sample(seed=key)             # [T*B, A]
            return a_flat.reshape((T_, B_, -1))                  # [T, B, A]
        
        def _img_step_all(rssm, prev_latent, actions):
            """prev_latent keys [T_eff,B,...], actions [T_eff,B,A] -> next_latent [T_eff,B,...]."""
            TB = prev_latent["deter"].shape[0] * prev_latent["deter"].shape[1]
            prev_flat = jax.tree_map(lambda x: x.reshape((TB,) + x.shape[2:]), prev_latent)
            a_flat    = actions.reshape((TB, -1))
            nxt_flat  = rssm.img_step(prev_flat, a_flat)
            return jax.tree_map(lambda x: x.reshape(prev_latent["deter"].shape[:2] + x.shape[1:]), nxt_flat)

        B = post["deter"].shape[1]
        perm = jax.random.permutation(nj.rng(), B)

        # Student current latents (paired within batch)
        s_i = post
        s_j = jax.tree_map(lambda x: x[:, perm], post)

        # Teacher current latents (only for sampling actions)
        t_i = teacher_post
        t_j = jax.tree_map(lambda x: x[:, perm], teacher_post)

        # Sample teacher actions under π_T for each branch
        k_i, k_j = jax.random.split(nj.rng())
        a_i = _sample_teacher_actions_TB(t_i, k_i)  # [T,B,A]
        a_j = _sample_teacher_actions_TB(t_j, k_j)  # [T,B,A]

        # Step the **student** dynamics with those actions
        s_i_next = _img_step_all(self.rssm, s_i, a_i)  # [T,B,...]
        s_j_next = _img_step_all(self.rssm, s_j, a_j)

        # φ-gaps (now and next)
        phi_now_i   = _phi_norm(s_i)        # [T,B,Dϕ] -> after norm L2 over last axis
        phi_now_j   = _phi_norm(s_j)
        phi_now_gap = jnp.linalg.norm(phi_now_i - phi_now_j, axis=-1)  # [T,B]

        # phi_next_i  = _phi_norm(s_i_next)
        # phi_next_j  = _phi_norm(s_j_next)
        # trans_gap   = jnp.linalg.norm(phi_next_i - phi_next_j, axis=-1)  # [T,B]

        # T, B = teacher_post["deter"].shape[:2]
        # t_feats = {**teacher_post, "embed": teacher_embed}
        # # r_pred = self.teacher_wm.heads["reward"](jax.lax.stop_gradient(t_feats)).mean()

        # # Normalize shape to [T, B] for downstream indexing
        # if r_pred.ndim == 3 and r_pred.shape[-1] == 1:
        #     r_pred = r_pred[..., 0]                      # [T, B]
        # # elif r_pred.ndim == 1:
        # #     r_pred = r_pred.reshape(T, B)                # [T*B] -> [T, B]
        # elif r_pred.ndim != 2:
        #     r_pred = r_pred.reshape(T, B)                # any odd case -> [T, B]

        # # Use jnp.take to avoid advanced-indexing pitfalls
        # r_pred_perm = jnp.take(r_pred, perm, axis=1)     # [T, B]
        # dr = jnp.abs(r_pred - r_pred_perm)               # [T, B]

        def huber(x, delta=1.0):
            a = jnp.abs(x)
            return jnp.where(a <= delta, 0.5 * x * x, delta * (a - 0.5 * delta))

        mode      = getattr(self.config, "bisim_mode", "single")   # "single" | "multi"
        K_actions = int(getattr(self.config, "bisim_K_actions", 1))
        H         = int(getattr(self.config, "bisim_ms_horizon", 1))
        ms_gamma  = float(getattr(self.config, "bisim_ms_discount", 0.99))

        T      = int(post["deter"].shape[0])
        H_eff  = max(1, min(H, T))
        T_eff  = T - (H_eff - 1)

       
        # def _rollout_phi_gap_piT(s_i0, s_j0, t_i_fix, t_j_fix, H_, key):
        #     """Closed-loop π_T: at each step sample actions from CURRENT teacher latents,
        #     step both teacher and student latents, accumulate discounted φ-gap."""
        #     # Slice to horizon-aligned prefix
        #     s_i = jax.tree_map(lambda x: x[:T_eff], s_i0)
        #     s_j = jax.tree_map(lambda x: x[:T_eff], s_j0)
        #     t_i = jax.tree_map(lambda x: x[:T_eff], t_i_fix)
        #     t_j = jax.tree_map(lambda x: x[:T_eff], t_j_fix)

        #     acc  = jnp.zeros(s_i["deter"].shape[:2], dtype=s_i["deter"].dtype)  # [T_eff,B]
        #     disc = 1.0
        #     k = key

        #     # Use a Python loop to avoid tracer leaks from any internal nj.rng().
        #     for _ in range(H_):
        #         k, k_i, k_j = jax.random.split(k, 3)

        #         # Actions from CURRENT teacher latents (closed-loop)
        #         a_i = _sample_teacher_actions_TB(t_i, k_i)      # [T_eff,B,A]
        #         a_j = _sample_teacher_actions_TB(t_j, k_j)      # [T_eff,B,A]

        #         # Step STUDENT latents (these feed φ)
        #         s_i = _img_step_all(self.rssm,            s_i, a_i)
        #         s_j = _img_step_all(self.rssm,            s_j, a_j)

        #         # Step TEACHER latents (to recondition π_T next step)
        #         t_i = _img_step_all(self.teacher_wm.rssm, t_i, a_i)
        #         t_j = _img_step_all(self.teacher_wm.rssm, t_j, a_j)

        #         disc = disc * ms_gamma
        #         acc  = acc + disc * jnp.linalg.norm(_phi_norm(s_i) - _phi_norm(s_j), axis=-1)  # [T_eff,B]
        #     return acc  # [T_eff,B]

        # if mode == "multi":
        #     if K_actions > 1:
        #         keys = jax.random.split(nj.rng(), K_actions)
        #         trans_gap_eff = jax.vmap(lambda k: _rollout_phi_gap_piT(s_i, s_j, t_i, t_j, H_eff, k))(keys).mean(axis=0)
        #     else:
        #         trans_gap_eff = _rollout_phi_gap_piT(s_i, s_j, t_i, t_j, H_eff, nj.rng())
        #     # Backfill trailing steps (where a lookahead of H won't fit) with one‑step φ-gap
        #     trans_gap = jnp.concatenate([trans_gap_eff, phi_now_gap[T_eff:]], axis=0)  # [T,B]
        # else:
        #     # single-step policy-conditional
        #     # trans_gap = w_deter * deter_gap_1 + w_stoch * stoch_gap_1  # [T,B]
        #     trans_gap   = jnp.linalg.norm(phi_next_i - phi_next_j, axis=-1) 

        # # Build the teacher bisimulation target
        # dT = alpha_r * dr + gamma_bisim * trans_gap                      # [T,B]


        # r_pred, dr, phi_next_i/j (student), trans_gap = ...
        # Teacher current latents (paired within batch)
        t_i = teacher_post
        t_j = jax.tree_map(lambda x: x[:, perm], teacher_post)

        # --- Transition term on TEACHER latents ---
        if mode == "multi":
            if K_actions > 1:
                keys = jax.random.split(nj.rng(), K_actions)
                trans_gap_eff = jax.vmap(lambda k: _rollout_phi_gap_piT_teacher(t_i, t_j, H_eff, k))(keys).mean(axis=0)
            else:
                trans_gap_eff = _rollout_phi_gap_piT_teacher(t_i, t_j, H_eff, nj.rng())
            # backfill trailing steps with a one‑step teacher φ‑gap
            phi_now_gap_T = jnp.linalg.norm(_phi_norm_T_no_grad(t_i) - _phi_norm_T_no_grad(t_j), axis=-1)
            trans_gap = jnp.concatenate([trans_gap_eff, phi_now_gap_T[T_eff:]], axis=0)  # [T,B]
        else:
            # single-step teacher-conditional
            k_i, k_j = jax.random.split(nj.rng())
            a_i = _sample_teacher_actions_TB(t_i, k_i)
            a_j = _sample_teacher_actions_TB(t_j, k_j)
            t_i_next = _img_step_all(self.teacher_wm.rssm, t_i, a_i)
            t_j_next = _img_step_all(self.teacher_wm.rssm, t_j, a_j)
            trans_gap = jnp.linalg.norm(_phi_norm_T_no_grad(t_i_next) - _phi_norm_T_no_grad(t_j_next), axis=-1)

        # --- Dense |ΔVᵀ| term via truncated teacher returns ---
        V_i_eff = _teacher_truncated_value(t_i, H_eff)  # [T_eff,B]
        V_j_eff = _teacher_truncated_value(t_j, H_eff)
        dV_eff = jnp.abs(V_i_eff - V_j_eff)             # [T_eff,B]
        # backfill tail with immediate teacher reward difference
        t_feats = {**teacher_post, "embed": teacher_embed}
        r_pred = self.teacher_wm.heads["reward"](jax.lax.stop_gradient(t_feats)).mean()
        if r_pred.ndim == 3 and r_pred.shape[-1] == 1:
            r_pred = r_pred[..., 0]
        r_pred_perm = jnp.take(r_pred, perm, axis=1)
        dV = jnp.concatenate([dV_eff, jnp.abs(r_pred - r_pred_perm)[T_eff:]], axis=0)  # [T,B]

        # --- Teacher-MDP bisim target ---
        # dT = alpha_r * dr + gamma_bisim * trans_gap
        dT = alpha_r * dV + gamma_bisim * trans_gap  # [T,B]

        mask_s = (1.0 - data["is_terminal"].astype(jnp.float32)) * (1.0 - data["is_first"].astype(jnp.float32))
        mask_t = (1.0 - teacher_data["is_terminal"].astype(jnp.float32)) * (1.0 - teacher_data["is_first"].astype(jnp.float32))
        if mask_s.ndim == 3 and mask_s.shape[-1] == 1:
            mask_s = mask_s.squeeze(-1)
        if mask_t.ndim == 3 and mask_t.shape[-1] == 1:
            mask_t = mask_t.squeeze(-1)
        mask = mask_s * mask_t  # [T,B]


        dT  = dT  * mask

        batch_mean = (jnp.sum(dT * mask) / (jnp.sum(mask) + 1e-6))
        # self.dT_ema = 0.99 * self.dT_ema + 0.01 * batch_mean
        new_ema = 0.99 * self.dT_ema.read() + 0.01 * batch_mean
        self.dT_ema.write(new_ema)                # or nj.assign(self.dT_ema, new_ema)
        # dT_norm = dT / (self.dT_ema.read() + 1e-6)
        dT_norm = jnp.clip(dT / (self.dT_ema.read() + 1e-6), a_min=0.0, a_max=self.config.get("bisim_target_clip", 10.0))


        pred = self.bisim_calib(phi_now_gap)
        err   = pred - jax.lax.stop_gradient(dT_norm)
        losses["bisim_pair"] = (huber(err, 1.0) * mask).mean()

        bisim_pair_loss = huber(phi_now_gap - jax.lax.stop_gradient(dT_norm), delta=1.0) * mask
                
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


        out = {"embed": embed, "post": post, "prior": prior, "teacher_embed": teacher_embed, "teacher_post": teacher_post, "teacher_prior": teacher_prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})

        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        teacher_last_latent = {k: v[:, -1] for k, v in teacher_post.items()}
        teacher_last_action = teacher_data["action"][:, -1]
        state = last_latent, last_action
        teacher_state = teacher_last_latent, teacher_last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization

        w = mask.astype(phi_now_gap.dtype)
        wsum = jnp.sum(w) + 1e-8

        x_mean = jnp.sum(w * phi_now_gap) / wsum
        y_mean = jnp.sum(w * dT) / wsum
        x_std  = jnp.sqrt(jnp.sum(w * (phi_now_gap - x_mean)**2) / wsum + 1e-8)
        y_std  = jnp.sqrt(jnp.sum(w * (dT      - y_mean)**2) / wsum + 1e-8)
        cov_xy = jnp.sum(w * (phi_now_gap - x_mean) * (dT - y_mean)) / wsum
        metrics["bisim_pair_corr"] = cov_xy / (x_std * y_std + 1e-8)

        metrics["bisim_phi_gap_mean"] = x_mean
        metrics["bisim_phi_gap_std"]  = x_std
        metrics["bisim_target_mean"]  = y_mean
        metrics["bisim_target_std"]   = y_std

        dtype = phi_now_gap.dtype
        metrics["bisim_mode"] = jnp.asarray(1.0 if mode == "multi" else 0.0, dtype)
        metrics["bisim_K"]    = jnp.asarray(float(K_actions), dtype)
        metrics["bisim_ms_H"] = jnp.asarray(float(H), dtype)

        scale_eff = jax.nn.softplus(self.bisim_calib.scale.read()) + 1e-6
        bias_eff  = jnp.maximum(self.bisim_calib.bias.read(), 0.0)
        metrics["bisim_scale"] = scale_eff
        metrics["bisim_bias"]  = bias_eff


        metrics.update(jaxutils_student.tensorstats(bisim_pair_loss, "bisim_pair"))  


        metrics.update({f"distill/{k}": v for k, v in distill.items()})
        return model_loss.mean(), (state,teacher_state, out, metrics)