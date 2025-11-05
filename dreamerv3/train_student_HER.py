import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    # config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        eval_name = name + "_test"
        print("Using eval task: ", eval_name)
        # env, env_config = car_dreamer.create_task(name, argv)
        # eval_env, eval_env_config = car_dreamer.create_task(eval_name, argv)

        raw_teacher_env, _ = car_dreamer.create_task(name, argv)
        raw_student_env, env_config = car_dreamer.create_task(name, argv)
        eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)

        config = config.update(env_config)
        eval_config = config.update(eval_env_config)

    config = embodied.Flags(config).parse(other)
    eval_config = embodied.Flags(eval_config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    from embodied.envs import from_gym

    teacher_gym = from_gym.TeacherObs(raw_teacher_env)
    student_gym = from_gym.StudentObs(raw_student_env)
    collect_gym  = from_gym.CollectorObs(raw_teacher_env)

    dreamerv3_config = config.dreamerv3
    # env = from_gym.FromGym(env)
    teacher_env = from_gym.FromGym(teacher_gym)
    student_env = from_gym.FromGym(student_gym)
    collect_env = from_gym.FromGym(collect_gym)


    # env = wrap_env(env, dreamerv3_config)
    teacher_env  = wrap_env(teacher_env, dreamerv3_config)
    student_env  = wrap_env(student_env, dreamerv3_config)
    collect_env  = wrap_env(collect_env, dreamerv3_config)

    # env = embodied.BatchEnv([env], parallel=False)

    teacher_env  = embodied.BatchEnv([teacher_env], parallel=False)
    student_env  = embodied.BatchEnv([student_env], parallel=False)
    collect_env  = embodied.BatchEnv([collect_env], parallel=False)

    eval_env = from_gym.FromGym(eval_env)
    eval_env = wrap_env(eval_env, dreamerv3_config)
    eval_env = embodied.BatchEnv([eval_env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    timer = embodied.Timer()

    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )

    teacher_agent = dreamerv3.agent_teacher(teacher_env.obs_space, teacher_env.act_space, step, dreamerv3_config)
    teacher_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")
    timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    timer.wrap("env", teacher_env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    def _warmup_teacher_modules(agent, obs_space):
        import numpy as np
        dummy = {k: np.zeros((1,) + v.shape, v.dtype) for k, v in obs_space.items()}
        for k in ("is_first", "is_last", "is_terminal"):
            if k in dummy:
                dummy[k][0] = (k == "is_first")
        agent.policy(dummy, None, mode="eval")

    _warmup_teacher_modules(teacher_agent, teacher_env.obs_space)

    expert = embodied.Checkpoint(logdir / "teacher.ckpt")
    timer.wrap("expert", expert, ["save", "load"])
    expert.step = step
    expert.agent = teacher_agent
    expert.replay = teacher_replay
    expert.load()  

    teacher_policy = lambda *args: teacher_agent.policy(*args, mode="eval")

    ###################################################################################################
    import numpy as np, hashlib

    def _warmup_teacher_modules(teacher_agent, obs_space):
        # Ensure WM variables exist
        dummy = {k: np.zeros((1,) + v.shape, v.dtype) for k, v in obs_space.items()}
        for k in ("is_first", "is_last", "is_terminal"):
            if k in dummy:
                dummy[k][0] = (k == "is_first")
        teacher_agent.policy(dummy, None, mode="eval")

    _warmup_teacher_modules(teacher_agent, teacher_env.obs_space)
    _state = teacher_agent.save()  # dict 'agent/...': ndarray or other leaves

    WM_PREFIXES     = ("agent/wm/enc/", "agent/wm/rssm/", "agent/wm/dec/", "agent/wm/rew/", "agent/wm/cont/")
    ACTOR_PREFIXES  = ("agent/task_behavior/ac/actor/",)
    # Exclude any non-parameter leaves
    EXCLUDE_FRAG    = ("/model_opt/", "/actor_opt/", "/critic_opt/", "/opt/",
                    "/updates/", "/good_steps/", "/grad_scale/", "/step/", "/state")

    _NUMERIC_KINDS = set("iufb")  # int, uint, float, bool

    def _iter_param_arrays(state, prefixes, exclude_frags):
        for k, v in state.items():
            if not any(k.startswith(p) for p in prefixes):
                continue
            if any(frag in k for frag in exclude_frags):
                continue
            # v should be an array; if not, skip
            try:
                arr = np.asarray(v)
            except Exception:
                continue
            if getattr(arr, "dtype", None) is None or arr.dtype.kind not in _NUMERIC_KINDS:
                continue
            yield k, arr

    def _fingerprint(state, prefixes):
        h = hashlib.sha256()
        l2_sum = 0.0
        n = 0
        for k, arr in _iter_param_arrays(state, prefixes, EXCLUDE_FRAG):
            l2_sum += float((arr * arr).sum())
            h.update(arr.tobytes())
            n += 1
        if n == 0:
            groups = sorted({"/".join(k.split("/")[:3]) + "/" for k in state})
            raise RuntimeError(f"No numeric params under {prefixes}. Groups present: {groups[:20]}")
        return float(np.sqrt(l2_sum)), h.hexdigest(), n

    teacher_wm_ref_l2,  teacher_wm_ref_sha,  wm_n    = _fingerprint(_state, WM_PREFIXES)
    teacher_act_ref_l2, teacher_act_ref_sha, actor_n = _fingerprint(_state, ACTOR_PREFIXES)

    print(f"[fingerprint] WM tensors: {wm_n}, L2={teacher_wm_ref_l2:.6g}")
    print(f"[fingerprint] ACTOR tensors: {actor_n}, L2={teacher_act_ref_l2:.6g}")

    dreamerv3_config = dreamerv3_config.update(dict(
        teacher_wm_ref_l2=teacher_wm_ref_l2,
        teacher_wm_ref_sha=teacher_wm_ref_sha,
        teacher_actor_ref_l2=teacher_act_ref_l2,
        teacher_actor_ref_sha=teacher_act_ref_sha,
    ))
    
    ###################################################################################################


    tp = teacher_agent.agent.task_behavior.ac.policy
    teacher_wm = teacher_agent.agent.wm
    agent = dreamerv3.agent_student(student_env.obs_space, student_env.act_space, teacher_wm, tp,  step, dreamerv3_config)  
    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")

    def her_goal_reward(ag, dg, infos, goal_radius=2.0, r_goal=200.0, r_collision=150.0, r_lane_inv=20.0, lane_cost_alpha=0.0, lane_cost_cap=1.5, is_first=None):
        # ag, dg: shape [T, 2]
        T = ag.shape[0]
        d = np.linalg.norm(ag - dg, axis=-1).astype(np.float32)               # [T]
        near = (d < goal_radius).astype(np.int32)
        r = (d < goal_radius).astype(np.float32) * r_goal  # [T]

        if is_first is None:
            is_first = np.zeros_like(T, dtype=bool)
            is_first[0] = True

        #Rising edge of "near", reset at episode starts.
        prev_near = np.roll(near, 1)
        prev_near[0] = 0
        success_first = (near == 1) & ((prev_near == 0) | is_first)

        r = success_first.astype(np.float32) * r_goal

        if isinstance(infos, (list, tuple)) and len(infos) == T:
            coll = np.array([float(info.get("collision", 0)) for info in infos], np.float32)
            inv  = np.array([float(i.get("lane_invasion", 0))  for i in infos], np.float32)

            r -= r_collision * coll
            r -= r_lane_inv * inv

            if lane_cost_alpha > 0.0:
                off = np.array([float(i.get("off_center_m", 0.0)) for i in infos], np.float32)
                off = np.clip(off, 0.0, lane_cost_cap)
                r -= lane_cost_alpha * off  # linear; switch to off**2 if you want quadratic


        return r.astype(np.float32)


        # # Optional: include collision penalty if provided in infos.
        # if isinstance(infos, (list, tuple)) and len(infos) == len(r):
        #     coll = np.array([float(info.get("collision", 0)) for info in infos], np.float32)
        #     r -= r_collision * coll
        # return r.astype(np.float32)

    reward_fn = her_goal_reward

    replay = embodied.replay.HERWrapper(replay, 
                                        reward_fn=reward_fn, 
                                        k=4,
                                        strategy="future",
                                        future_horizon=None,
                                        ag_key="achieved_goal",
                                        dg_key="desired_goal",
                                        reward_key="reward",
                                        strict_future=True
                                        )
    
    eval_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "eval_replay")  
    timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    timer.wrap("env", student_env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

   


    embodied.run.train_student(agent, teacher_policy, student_env, eval_env, replay, eval_replay, teacher_replay, logger, args)


if __name__ == "__main__":
    main()
