import copy, numpy as np, random

class HERWrapper:
    def __init__(self, base_replay, reward_fn, k=4, strategy="future", future_horizon=50,
                 ag_key="achieved_goal", dg_key="desired_goal", reward_key="reward" , strict_future=True):
        self.base = base_replay
        self.reward_fn = reward_fn         # expects (ag[T,2], dg[T,2], info_dict_list) -> r[T]
        self.k = k
        self.strategy = strategy
        self.future_horizon = future_horizon
        self.ag_key, self.dg_key, self.reward_key = ag_key, dg_key, reward_key
        self.strict_future = strict_future

        
    def add(self, *args, **kwargs):
        return self.base.add(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.base.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        # if your base replay exposes load()
        return getattr(self.base, "load")(*args, **kwargs)

    def __len__(self):
        return len(self.base)

    @property
    def stats(self):
        return self.base.stats

    def prioritize(self, *args, **kwargs):
        if hasattr(self.base, "prioritize"):
            return self.base.prioritize(*args, **kwargs)

    def update_visit_count(self, *args, **kwargs):
        if hasattr(self.base, "update_visit_count"):
            return self.base.update_visit_count(*args, **kwargs)

    def __getattr__(self, name):
        # fallback for any other attributes the framework may access
        return getattr(self.base, name)
    


    
    def dataset(self):
        rng = np.random.default_rng()
        for seq in self.base.dataset():
            # 1) yield original
            yield seq

            # 2) derive per-step arrays
            T = int(len(seq[self.reward_key]))

            # --- robust lookup for achieved_goal ---
            ag_field = self.ag_key
            ag_src = seq.get(ag_field, None)

            if ag_src is None:
                # Try un-prefixed / prefixed variants
                if ag_field.startswith("info/"):
                    alt = ag_field[len("info/"):]     # "achieved_goal"
                else:
                    alt = "info/" + ag_field          # "info/achieved_goal"
                ag_src = seq.get(alt, None)

            # If we still didn't find anything, skip HER augmentation for this sequence.
            if ag_src is None:
                # We already yielded the original seq above; just continue.
                continue

            ag = np.asarray(ag_src, np.float32)[:T]   # [T, 2]
            # ag = np.asarray(seq[self.ag_key], np.float32)[:T]   # [T, 2]
            # coll = np.asarray(seq.get("collision",
            #                         seq.get("info/collision", np.zeros(T))),
            #                 np.float32)[:T]
            # infos = [{"collision": bool(c)} for c in coll]

            coll = np.asarray(seq.get("info/collision_step", np.zeros(T)),np.float32)[:T]
            inv  = np.asarray(seq.get("info/lane_invasion_step", np.zeros(T)), np.float32)[:T]
            off  = np.asarray(seq.get("info/off_center_m",       np.zeros(T)), np.float32)[:T]  # optional

            infos = [
            {"collision": bool(c), "lane_invasion": bool(i), "off_center_m": float(o)}
            for c, i, o in zip(coll, inv, off)
            ]
            # infos = [{"collision": bool(c)} for c in coll]

            is_first = np.asarray(seq.get("is_first", np.zeros(T)), bool)[:T]

            def _same_episode_span(is_first):
                T = len(is_first)
                starts = np.flatnonzero(is_first.astype(bool))
                if len(starts) == 0 or starts[0] != 0:
                    starts = np.r_[0, starts]
                starts = np.r_[starts, T]
                next_start = np.empty(T, np.int64)
                for a, b in zip(starts[:-1], starts[1:]):
                    next_start[a:b] = b
                idx = np.arange(T, dtype=np.int64)
                return (next_start - 1) - idx


            
            span_ce = _same_episode_span(is_first)
            if self.future_horizon is None:
                span = span_ce.astype(np.int64)                    # full remainder
            else:
                span = np.minimum(self.future_horizon, np.maximum(0, span_ce)).astype(np.int64)
            # span = np.minimum(self.future_horizon, np.maximum(0, span_ce)).astype(np.int64)

            # idx = np.arange(T)
            # span = np.minimum(self.future_horizon, (T - 1) - idx)

            idx  = np.arange(T, dtype=np.int64)
            # span = np.minimum(self.future_horizon, (T - 1) - idx).astype(np.int64)  # [T]

            

            for _ in range(self.k):
                # 3) choose relabeling strategy
                if self.strategy == "future":
                    offs = np.zeros(T, dtype=np.int64)
                    mask = span > 0
                    if mask.any():
                        # strictly future: 1..span[i]
                        # option A (safe across NumPy versions):
                        offs[mask] = 1 + np.floor(rng.random(mask.sum()) * span[mask]).astype(np.int64)
                        # option B (also fine on recent NumPy):
                        # offs[mask] = 1 + rng.integers(0, span[mask], size=mask.sum())
                    elif self.strict_future:
                        # no valid future steps to sample from
                        continue

                    j  = idx + offs  # last step keeps j[i]=i
                    dg = ag[j]
                    # offs = rng.integers(1, span + 1, size=T)     # strictly future
                    # offs = np.where(span > 0, offs, 0)
                    # j = idx + offs
                    # dg = ag[j]
                elif self.strategy == "final":
                    dg = np.repeat(ag[None, T - 1, :], T, axis=0)
                elif self.strategy in ("episode", "random"):
                    j = rng.integers(0, T)
                    dg = np.repeat(ag[None, j, :], T, axis=0)
                else:
                    raise ValueError(f"Unknown strategy {self.strategy}")

                # 4) write only changed fields
                # seq2 = dict(seq)
                # seq2[self.dg_key] = dg.astype(np.float32)

                # # --- robust lookup for desired_goal ---
                # dg_field = self.dg_key
                # dg_src = seq.get(dg_field, None)

                # if dg_src is None:
                #     if dg_field.startswith("info/"):
                #         alt = dg_field[len("info/"):]     # "desired_goal"
                #     else:
                #         alt = "info/" + dg_field          # "info/desired_goal"
                #     dg_src = seq.get(alt, None)

                # # If still missing, fall back to using the same points as achieved_goal
                # if dg_src is None:
                #     dg_src = ag_src

                # dg = np.asarray(dg_src, np.float32)[:T]


                # seq2[self.reward_key] = np.asarray(
                #     self.reward_fn(ag, dg, infos, is_first=np.asarray(seq.get("is_first", np.zeros(T)), bool)[:T]), np.float32
                # )[:T]

                # 4) write only changed fields
                seq2 = dict(seq)

                # dg here is the HER-sampled desired goal from above
                dg_her = dg.astype(np.float32)        # [T, 2]
                seq2[self.dg_key] = dg_her

                # use the same is_first you computed earlier
                seq2[self.reward_key] = np.asarray(
                    self.reward_fn(ag, dg_her, infos, is_first=is_first),
                    np.float32
                )[:T]



                # If learner consumes these, stop bootstrapping after success.
                succ_idx = np.flatnonzero(seq2[self.reward_key] > 0)
                if succ_idx.size and "discount" in seq:
                    cut = succ_idx[0]
                    disc = np.array(seq["discount"][:T], np.float32)
                    disc[cut:] = 0.0
                    seq2["discount"] = disc
                if succ_idx.size and "is_terminal" in seq:
                    term = np.array(seq["is_terminal"][:T], bool)
                    term[succ_idx[0]] = True
                    seq2["is_terminal"] = term

                yield seq2


    # def dataset(self):
    #     rng = np.random.default_rng()
    #     for seq in self.base.dataset():
    #         # 1) yield original sequence
    #         yield seq

    #         T = len(seq[self.reward_key])
    #         ag = np.asarray(seq[self.ag_key], dtype=np.float32)[:T]     # [T,2]
    #         # Prepare per-step infos from existing fields if present
    #         coll = np.asarray(seq.get("collision",
    #                                   seq.get("info/collision", np.zeros(T))),
    #                                 np.float32)[:T]
    #         infos = [{"collision": bool(c)} for c in coll]

    #         idx = np.arange(T)
    #         span = np.minimum(self.future_horizon, (T - 1) - idx)

    #         for _ in range(self.k):
    #             # 2) copy the sequence
    #             seq2 = {k: (v.copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
    #                     for k, v in seq.items()}

    #             # 3) sample new desired goals
    #             if self.strategy == "future":
    #                 idx = np.arange(T)
    #                 span = np.minimum(self.future_horizon, (T - 1) - idx)
    #                 j = idx + np.random.randint(0, span + 1)
    #                 dg = ag[j]
    #             elif self.strategy == "final":
    #                 dg = np.repeat(ag[None, -1, :], T, axis=0)
    #             elif self.strategy == "episode":
    #                 j = np.random.randint(0, T)
    #                 dg = np.repeat(ag[None, j, :], T, axis=0)
    #             else:  # random
    #                 j = np.random.randint(0, T)
    #                 dg = np.repeat(ag[None, j, :], T, axis=0)

    #             # 4) overwrite goal and reward
    #             seq2[self.dg_key] = dg.astype(np.float32)
    #             seq2[self.reward_key] = np.asarray(self.reward_fn(ag, dg, infos), dtype=np.float32)

    #             yield seq2
