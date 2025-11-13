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

        env, env_config = car_dreamer.create_task(name, argv)
        eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)

        config = config.update(env_config)
        eval_config = config.update(eval_env_config)

    config = embodied.Flags(config).parse(other)
    eval_config = embodied.Flags(eval_config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)

    # -------------------- student counter + logger --------------------
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

    timer = embodied.Timer()

    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )


    teacher_step = embodied.Counter()  # separate counter for teacher
    teacher_agent = dreamerv3.agent_teacher(env.obs_space, env.act_space, teacher_step, dreamerv3_config)
    teacher_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")
    timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    expert = embodied.Checkpoint(logdir / "teacher.ckpt")
    timer.wrap("expert", expert, ["save", "load"])
    expert.step = teacher_step
    expert.agent = teacher_agent
    expert.replay = teacher_replay
    expert.load()  

    teacher_policy = lambda *args: teacher_agent.policy(*args, mode="eval")

    tp = teacher_agent.agent.task_behavior.ac.policy
    teacher_wm = teacher_agent.agent.wm
    agent = dreamerv3.agent_student_bisim(env.obs_space, env.act_space, teacher_wm, tp,  step, dreamerv3_config)  

    # # ----------------------------------------
    # # Copy teacher world model parameters into the student's teacher_wm subtree
    # # ----------------------------------------

    # # 1) Dump parameter trees
    # teacher_vars = teacher_agent.save()   # tree: { "agent/wm/...": array, ... }
    # student_vars = agent.save()           # tree: { "agent/...", "agent/teacher_wm/...", ... }

    # def copy_teacher_wm_into_student(teacher_vars, student_vars):
    #     teacher_prefix = "agent/wm/"
    #     student_prefix = "agent/teacher_wm/"

    #     num_copied = 0

    #     for k, v in teacher_vars.items():
    #         if not k.startswith(teacher_prefix):
    #             continue
    #         suffix = k[len(teacher_prefix):]           # path inside wm
    #         dst_key = student_prefix + suffix          # corresponding key under teacher_wm

    #         if dst_key in student_vars:
    #             student_vars[dst_key] = v
    #             num_copied += 1
    #         else:
    #             # Optional: debug print if you'd like to sanity check
    #             print(f"[WARN] No matching key in student for teacher key {k}")

    #     # Optional: sanity check
    #     # print(f"[INFO] Copied {num_copied} teacher_wm parameters into student.teacher_wm")
    #     return student_vars

    # student_vars = copy_teacher_wm_into_student(teacher_vars, student_vars)

    # # 2) Load back into student agent and sync to devices
    # agent.load(student_vars)
    # agent.sync()   # if your JAXAgent has sync() for multi-device; no-op otherwise

    # t_vars = teacher_agent.save()
    # s_vars = agent.save()

    # def subtree_norm(vars, prefix):
    #     import jax.numpy as jnp
    #     arrays = [v.reshape(-1) for k, v in vars.items() if k.startswith(prefix)]
    #     return float(jnp.linalg.norm(jnp.concatenate(arrays))) if arrays else 0.0

    # teacher_wm_norm  = subtree_norm(t_vars, "agent/wm/")
    # student_tw_norm  = subtree_norm(s_vars, "agent/teacher_wm/")

    # print("teacher wm  norm:", teacher_wm_norm)
    # print("student teacher_wm norm:", student_tw_norm)



    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    eval_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "eval_replay")  
    
    embodied.run.train_student(agent, teacher_policy, env, eval_env, replay, eval_replay, teacher_replay, logger, args)


if __name__ == "__main__":
    main()
