import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

import csv, atexit

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
        env, env_config = car_dreamer.create_task(name, argv)
        eval_env, eval_env_config = car_dreamer.create_task(eval_name, argv)
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

    dreamerv3_config = config.dreamerv3
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    eval_env = from_gym.FromGym(eval_env)
    eval_env = wrap_env(eval_env, dreamerv3_config)
    eval_env = embodied.BatchEnv([eval_env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    agent = dreamerv3.agent_teacher(env.obs_space, env.act_space, step, dreamerv3_config)

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

    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )

    embodied.run.train_teacher(agent, env, eval_env, replay, eval_replay, logger, args)


if __name__ == "__main__":
    main()
