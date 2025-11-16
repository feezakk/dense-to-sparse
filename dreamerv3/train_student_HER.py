import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

import numpy as np

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

        # raw_teacher_env, _ = car_dreamer.create_task(name, argv)
        # raw_student_env, env_config = car_dreamer.create_task(name, argv)
        # eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)

        env, env_config = car_dreamer.create_task(name, argv)
        eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)

        config = config.update(env_config)
        eval_config = config.update(eval_env_config)

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

    # teacher_gym = from_gym.TeacherObs(raw_teacher_env)
    # student_gym = from_gym.StudentObs(raw_student_env)
    # collect_gym  = from_gym.CollectorObs(raw_teacher_env)

    dreamerv3_config = config.dreamerv3

    shared_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")

    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    eval_env = from_gym.FromGym(eval_env)
    eval_env = wrap_env(eval_env, dreamerv3_config)
    eval_env = embodied.BatchEnv([eval_env], parallel=False)

    # env = from_gym.FromGym(env)
    # teacher_env = from_gym.FromGym(teacher_gym)
    # student_env = from_gym.FromGym(student_gym)
    # collect_env = from_gym.FromGym(collect_gym)


    # # env = wrap_env(env, dreamerv3_config)
    # teacher_env  = wrap_env(teacher_env, dreamerv3_config)
    # student_env  = wrap_env(student_env, dreamerv3_config)
    # collect_env  = wrap_env(collect_env, dreamerv3_config)

    # # env = embodied.BatchEnv([env], parallel=False)

    # teacher_env  = embodied.BatchEnv([teacher_env], parallel=False)
    # student_env  = embodied.BatchEnv([student_env], parallel=False)
    # collect_env  = embodied.BatchEnv([collect_env], parallel=False)

    # eval_env = from_gym.FromGym(eval_env)
    # eval_env = wrap_env(eval_env, dreamerv3_config)
    # eval_env = embodied.BatchEnv([eval_env], parallel=False)

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

    teacher_step = embodied.Counter()  # separate counter for teacher
    teacher_agent = dreamerv3.agent_teacher(env.obs_space, env.act_space, teacher_step, dreamerv3_config)
    teacher_replay = shared_replay
    timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    # teacher_step = embodied.Counter()  # separate counter for teacher
    # teacher_agent = dreamerv3.agent_teacher(teacher_env.obs_space, teacher_env.act_space, teacher_step, dreamerv3_config)
    # teacher_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")
    # timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    # timer.wrap("env", teacher_env, ["step"])
    # timer.wrap("replay", teacher_replay, ["add", "save"])
    # timer.wrap("logger", logger, ["write"])

    expert = embodied.Checkpoint(logdir / "teacher.ckpt")
    timer.wrap("expert", expert, ["save", "load"])
    expert.step = step
    expert.agent = teacher_agent
    expert.replay = teacher_replay
    expert.load()  

    teacher_policy = lambda *args: teacher_agent.policy(*args, mode="eval")

    tp = teacher_agent.agent.task_behavior.ac.policy
    teacher_wm = teacher_agent.agent.wm
    agent = dreamerv3.agent_student_bisim(env.obs_space, env.act_space, teacher_wm, tp,  step, dreamerv3_config)  

    # agent = dreamerv3.agent_student(student_env.obs_space, student_env.act_space, teacher_wm, tp,  step, dreamerv3_config)  

    import jax
    import jax.numpy as jnp
    import numpy as np

    teacher_vars = teacher_agent.save()
    student_vars = agent.save()

    copied = 0
    missing = 0
    shape_mismatch = 0

    for k, v in teacher_vars.items():
        if not k.startswith("agent/wm/"):
            continue
        k_student = k.replace("agent/wm/", "agent/teacher_wm/")

        if k_student not in student_vars:
            print("[MISSING IN STUDENT]", k_student)
            missing += 1
            continue

        if student_vars[k_student].shape != v.shape:
            print("[SHAPE MISMATCH]", k_student,
                "student", student_vars[k_student].shape,
                "teacher", v.shape)
            shape_mismatch += 1
            continue

        student_vars[k_student] = v
        copied += 1

    print(f"[WM copy] Copied {copied} tensors, missing={missing}, shape_mismatch={shape_mismatch}")
    agent.load(student_vars)

    print("***************************************************************")
    print(f"[WM copy] Copied {copied} teacher WM tensors into student WM.")
    print("***************************************************************")
    
    def param_norm(agent_obj, label):
        vars_dict = agent_obj.save()

        wm_arrays = [
            v
            for k, v in vars_dict.items()
            if k.startswith("agent/wm/") and isinstance(v, (np.ndarray, jnp.ndarray))
        ]

        if not wm_arrays:
            print(f"[{label}] No agent/wm/ parameter arrays found.")
            return

        # Explicitly move to host to avoid implicit device->host transfer
        host_arrays = [np.ravel(jax.device_get(v)) for v in wm_arrays]
        flat = np.concatenate(host_arrays)
        norm = np.linalg.norm(flat)

        print(f"[{label}] ||agent/wm||_2 = {norm:.6e}")

    param_norm(teacher_agent, "Teacher")
    param_norm(agent, "Student (after copy)")

    def wm_subtree_norm(vars_dict, prefix):
        arrs = [v for k, v in vars_dict.items()
                if k.startswith(prefix) and isinstance(v, (np.ndarray, jnp.ndarray))]
        if not arrs:
            print(f"[{prefix}] No params found")
            return
        flat = np.concatenate([np.ravel(v) for v in arrs])
        print(f"[{prefix}] L2 norm = {np.linalg.norm(flat):.6e}")

    teacher_vars  = teacher_agent.save()
    student_vars  = agent.save()

    wm_subtree_norm(teacher_vars, "agent/wm/")
    wm_subtree_norm(student_vars, "agent/teacher_wm/")

    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")

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
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    # replay = shared_replay
    # eval_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "eval_replay")  
    

    nonzeros = set()

   

    embodied.run.train_student_bisim(agent, teacher_policy, env, eval_env, replay, eval_replay, teacher_replay, logger, args)


if __name__ == "__main__":
    main()
