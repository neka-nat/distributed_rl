from collections import deque
import numpy as np
from gym import spaces
from gym.envs.atari.atari_env import AtariEnv
from libs import utils

class MultiFrameAtariEnv(AtariEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}
    no_op_steps = 30
    def __init__(self, game='pong', obs_type='image', buf_size=4, gray=True,
                 frameskip=4, repeat_action_probability=0.):
        super(MultiFrameAtariEnv, self).__init__(game, obs_type,
                                                 frameskip, repeat_action_probability)
        self._cur_st = None
        self._nx_st = None
        self._img_buf = deque(maxlen=buf_size)
        self._gray = gray
        self._shape = (84, 84)
        if self._gray:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self._shape[0], self._shape[1], buf_size),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self._shape[0], self._shape[1], 3, buf_size),
                                                dtype=np.uint8)
        self._initialize()

    def _initialize(self):
        self._nx_st = super(MultiFrameAtariEnv, self).reset()
        for _ in range(self._img_buf.maxlen):
            self._img_buf.append(utils.preprocess(self._nx_st, self._shape, self._gray))
        for _ in range(np.random.randint(1, self.no_op_steps) // self.frameskip):
            self.step(0)

    def step(self, a):
        self._cur_st = self._nx_st.copy()
        self._nx_st, reward, done, info = super(MultiFrameAtariEnv, self).step(a)
        nx_st = np.maximum(self._nx_st, self._cur_st) if self._gray else self._nx_st
        self._img_buf.append(utils.preprocess(nx_st, self._shape, self._gray))
        return np.array(list(self._img_buf)), reward, done, info

    def reset(self):
        self._img_buf.clear()
        self._initialize()
        return np.array(list(self._img_buf))

from gym.envs.registration import register

register(
    id='MultiFramePong-v0',
    entry_point='libs.wrapped_env:MultiFrameAtariEnv',
    kwargs={'obs_type': 'image'},
    max_episode_steps=10000,
    nondeterministic=False,
)

register(
    id='MultiFrameBreakout-v0',
    entry_point='libs.wrapped_env:MultiFrameAtariEnv',
    kwargs={'game': 'breakout', 'obs_type': 'image'},
    max_episode_steps=10000,
    nondeterministic=False,
)

register(
    id='SingleFrameBreakout-v0',
    entry_point='libs.wrapped_env:MultiFrameAtariEnv',
    kwargs={'game': 'breakout', 'obs_type': 'image', 'buf_size': 1, 'gray': False},
    max_episode_steps=10000,
    nondeterministic=False,
)

register(
    id='MultiFrameSpaceInvaders-v0',
    entry_point='libs.wrapped_env:MultiFrameAtariEnv',
    kwargs={'game': 'space_invaders', 'obs_type': 'image'},
    max_episode_steps=10000,
    nondeterministic=False,
)
