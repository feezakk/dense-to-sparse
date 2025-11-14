import collections

import numpy as np

from .basics import convert


class DriverStudentBisim:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        # self._env_steps = np.zeros(len(self._env), np.int64)
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._acts = {k: convert(np.zeros((len(self._env),) + v.shape, v.dtype)) for k, v in self._env.act_space.items()}
        self._acts["reset"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._eps_info = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    # def __call__(self, student_policy, steps=0, episodes=0):
    def __call__(self, student_policy, teacher_policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(student_policy, teacher_policy, step, episode)

    def _step(self, student_policy, teacher_policy, step, episode):
        assert all(len(x) == len(self._env) for x in self._acts.values())
        # print("self._acts", self._acts)
        acts = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
        # print("Driver acts", acts)
        obs, info = self._env.step(acts)
        obs = {k: convert(v) for k, v in obs.items()}
        info = {k: convert(v) for k, v in info.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        acts, self._state = student_policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}

        # print("driver obs", obs.keys())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # if teacher_policy is not None:
        #     # 3) (NEW) Also call the TEACHER for labeling, but do NOT apply to env
        #     teacher_acts, _ = teacher_policy(obs)  # or teacher_acts = teacher_policy(obs) if stateless
        #     teacher_acts = {f"teacher_{k}": convert(v) for k, v in teacher_acts.items()}
        #     # print("******************" , teacher_acts)
        #     # print("******************" , acts)
        # else:
        #     teacher_acts = {}



        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()
        self._acts = acts
        # trns = {**obs, **acts, **teacher_acts}

        trns = {**obs, **acts}


        # # Combine everything into transition:
        # if teacher_policy is None:
        #     trns = {**obs, **acts}
        # else:
        #     trns = {**obs, **acts, **teacher_acts}

        if obs["is_first"].any():
            for i, first in enumerate(obs["is_first"]):
                if first:
                    self._eps[i].clear()
                    self._eps_info[i].clear()
        for i in range(len(self._env)):
            # print("***************", trns.keys())
            trn = {k: v[i] for k, v in trns.items()}
            # print("***************", info.keys())
            # self._env_steps[i] += 1
            # trn["env_step"] = self._env_steps[i]
            inf = {k: v[i] for k, v in info.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [self._eps_info[i][k].append(v) for k, v in inf.items()]
            [fn(trn, inf, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if obs["is_last"].any():
            for i, done in enumerate(obs["is_last"]):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    ep_info = {k: convert(v) for k, v in self._eps_info[i].items()}
                    [fn(ep.copy(), ep_info.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value