import re
import warnings

import embodied
import numpy as np
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

def eval_only(agent, env, logger, args):
    print("Start evaluation.")
    print("args:", args)
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    step = logger.step
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    # ---------- CSV writers (NEW) ----------
    def _make_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index","env_step","length","return",
            "success","collision","time_exceeded","not_moving","past_goal",
            # --- NEW metrics ---
            "distance_m", "distance_km",
            "collisions_per_km", "lane_invasions", "lane_invasions_per_km",
            "exceed", "returned", "overtaken",
            "min_ego_lead_dist", "mean_off_center_m",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader(); f.flush()
        return f, w

    train_csv_f, train_csv_w = _make_writer(logdir / "train_success.csv")
    eval_csv_f,  eval_csv_w  = _make_writer(logdir / "eval_success.csv")
    atexit.register(lambda: (train_csv_f.close(), eval_csv_f.close()))
    train_ep_idx = {"v": 0}
    eval_ep_idx  = {"v": 0}

    def _csv_log(ep, ep_info, is_eval=False):
        # Episode stats
        length = int(len(ep["reward"]) - 1)
        ret = float(ep["reward"].astype(np.float64).sum())
        def _any(k):  # handles missing keys
            v = ep_info.get(k, [])
            return bool(np.any(np.array(v)))
        

        # --- NEW: helper arrays with safe defaults ---
        step_dist_arr = np.array(ep_info.get("step_distance", [0.0]), dtype=np.float64)
        off_center_arr = np.array(ep_info.get("off_center_m", [0.0]), dtype=np.float64)
        lane_inv_arr = np.array(ep_info.get("lane_invasion_step", [0]), dtype=np.int32)
        ego_lead_arr = np.array(ep_info.get("ego_lead_dist", [np.inf]), dtype=np.float64)

        distance_m = float(step_dist_arr.sum())
        distance_km = distance_m / 1000.0 if distance_m > 0 else 0.0
        lane_invasions = int(lane_inv_arr.sum())
        mean_off_center_m = float(off_center_arr.mean()) if off_center_arr.size else 0.0
        min_ego_lead_dist = float(ego_lead_arr.min()) if ego_lead_arr.size else np.inf

        # episode-level flags
        success = int(_any("goal_reached"))          # or "overtake" if you prefer
        collision = int(_any("collision"))
        time_exceeded = int(_any("time_exceeded"))
        not_moving = int(_any("not_moving"))
        past_goal = int(_any("past_goal"))

        exceed = int(_any("exceed"))
        returned = int(_any("returned"))
        overtaken = int(_any("overtake"))

        eps = 1e-6
        collisions_per_km = collision / max(distance_km, eps)
        lane_inv_per_km = lane_invasions / max(distance_km, eps)


        row = {
            "episode_index": (eval_ep_idx["v"] if is_eval else train_ep_idx["v"]),
            "env_step": int(logger.step),
            "length": length,
            "return": ret,
            "success": int(_any("goal_reached")),
            "collision": int(_any("collision")),
            "time_exceeded": int(_any("time_exceeded")),
            "not_moving": int(_any("not_moving")),
            "past_goal": int(_any("past_goal")),
            # --- NEW metrics ---
            "distance_m": distance_m,
            "distance_km": distance_km,
            "collisions_per_km": collisions_per_km,
            "lane_invasions": lane_invasions,
            "lane_invasions_per_km": lane_inv_per_km,
            "exceed": exceed,
            "returned": returned,
            "overtaken": overtaken,
            "min_ego_lead_dist": min_ego_lead_dist,
            "mean_off_center_m": mean_off_center_m,
        }
        w, f = (eval_csv_w, eval_csv_f) if is_eval else (train_csv_w, train_csv_f)
        w.writerow(row); f.flush()
        if is_eval: eval_ep_idx["v"] += 1
        else:       train_ep_idx["v"] += 1
    # ---------------------------------------

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        def log(key, value):
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = value.sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = value.mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = value.max(0).mean()

        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            log(key, value)
        for key, value in ep_info.items():
            log(key, value)

        logger.add(metrics.result())
        logger.add(timer.stats(), prefix="timer")
        logger.write(fps=True)

        metrics.add(stats, prefix="stats")

    def per_step(tran):
        step.increment()

    driver = embodied.DriverTeacher(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_step(lambda tran, info, _: per_step(step))
    driver.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=True))

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified.")

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        driver(policy, steps=100)
    logger.write()


def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    # config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)

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

    dreamerv3_config = dreamerv3_config.update(
        {
            "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
            "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.steps": 5e4,
        }
    )

    agent = dreamerv3.agent_teacher(env.obs_space, env.act_space, step, dreamerv3_config)
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
    )
    eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
