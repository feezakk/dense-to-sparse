import re

import embodied
import jax
import numpy as np
import csv, atexit


def train(agent, env, eval_env, replay, eval_replay, logger, args):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    # ---------- CSV writers (NEW) ----------
    def _make_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index","env_step","length","return",
            "success","collision","time_exceeded","not_moving","past_goal"
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
        }
        w, f = (eval_csv_w, eval_csv_f) if is_eval else (train_csv_w, train_csv_f)
        w.writerow(row); f.flush()
        if is_eval: eval_ep_idx["v"] += 1
        else:       train_ep_idx["v"] += 1
    # ---------------------------------------

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        success = float(np.any(np.array(ep_info.get("goal_reached", [0]))))
        collision = float(np.any(np.array(ep_info.get("collision", [0]))))
        past_goal = float(np.any(np.array(ep_info.get("past_goal", [0]))))
        not_moving = float(np.any(np.array(ep_info.get("not_moving", [0]))))
        time_exceeded = float(np.any(np.array(ep_info.get("time_exceeded", [0]))))

        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
                "success": success,
                "collision": collision,
                "past_goal": past_goal,
                "not_moving": not_moving,
                "time_exceeded": time_exceeded
            },
            prefix="episode",
        )
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.DriverTeacher(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=False))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    driver_eval = embodied.DriverTeacher(eval_env)
    driver_eval.on_step(lambda trn, inf, i, **kw: eval_replay.add(trn, worker=i))
    driver_eval.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver_eval.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=True))

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    print("Prefill eval dataset.")
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=100)

    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    dataset_eval = agent.dataset(eval_replay.dataset)


    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(_, __, ___):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            if getattr(replay, "update_visit_count", False):
                replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(replay.stats, prefix="report")
            with timer.scope("dataset_eval"):
                eval_batch = next(dataset_eval)
            logger.add(agent.report(eval_batch), prefix="eval")
            logger.add(replay.stats, prefix="replay")
            logger.add(eval_replay.stats, prefix="eval_replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print("Start training loop.")
    driver._state = None
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    policy_eval = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        if should_eval(step):
            print("Starting evaluation at step", int(step))
            driver_eval.reset()
            driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
