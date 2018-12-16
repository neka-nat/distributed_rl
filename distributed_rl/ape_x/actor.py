# -*- coding: utf-8 -*-
import numpy as np
from itertools import count
from collections import deque
import redis
import torch
from ..libs import replay_memory, utils

class Actor(object):
    """Actor of Ape-X

    Args:
        name (str): name of the actor process's name
        env (gym.Env): Open AI gym environment object
        policy_net (torch.nn.Module): Q-function network
        vis (visdom.Visdom): visdom object
        hostname (str, optional): host name of redis server
        batch_size (int, optional): batch data size when sending to learner
        nstep_return (int, optional): number of multi-step boostrapping
        gamma (float, optional): discount factor
        clip (function, optional): reward clipping function
        target_update (int, optional): update frequency of the target network
        eps_decay (int, optional): Decay of random action rate in e-greedy
        device (torch.device, optional): calculation device
    """
    EPS_BASE = 0.4
    EPS_ALPHA = 7.0
    def __init__(self, name, env, policy_net, vis, hostname='localhost',
                 batch_size=50, nstep_return=3, gamma=0.999,
                 clip=lambda x: min(max(-1.0, x), 1.0),
                 target_update=200, eps_decay=10000000,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self._env = env
        self._name = name
        self._vis = vis
        self._batch_size = batch_size
        self._nstep_return = nstep_return
        self._gamma = gamma
        self._clip = clip
        self._target_update = target_update
        self._eps_decay = eps_decay
        self._policy_net = policy_net
        self._policy_net.eval()
        self._device = device
        self._win1 = self._vis.image(utils.preprocess(self._env.env._get_image()))
        self._win2 = self._vis.line(X=np.array([0]), Y=np.array([0.0]),
                                    opts=dict(title='Score %s' % self._name))
        self._local_memory = replay_memory.ReplayMemory(1000)
        self._connect = redis.StrictRedis(host=hostname)

    def _pull_params(self):
        params = self._connect.get('params')
        if not params is None:
            print("[%s] Sync params." % self._name)
            self._policy_net.load_state_dict(utils.loads(params))

    def run(self):
        state = self._env.reset()
        step_buffer = deque(maxlen=self._nstep_return)
        gamma_nsteps = [self._gamma ** i for i in range(self._nstep_return + 1)]
        sum_rwd = 0
        n_episode = 0
        for t in count():
            # Select and perform an action
            eps = self.EPS_BASE ** (1.0 + t / (self._eps_decay - 1.0) * self.EPS_ALPHA)
            action = utils.epsilon_greedy(torch.from_numpy(state).unsqueeze(0).to(self._device),
                                          self._policy_net, eps)
            next_state, reward, done, _ = self._env.step(action.item())
            reward = torch.tensor([self._clip(reward)])
            done = torch.tensor([float(done)])
            step_buffer.append(utils.Transition(torch.from_numpy(state), action, reward,
                                                torch.from_numpy(next_state), done))
            if len(step_buffer) == step_buffer.maxlen:
                r_nstep = sum([gamma_nsteps[-(i + 2)] * step_buffer[i].reward for i in range(step_buffer.maxlen)])
                self._local_memory.push(utils.Transition(step_buffer[0].state, step_buffer[0].action, r_nstep,
                                                         step_buffer[-1].next_state, step_buffer[-1].done))
            self._vis.image(utils.preprocess(self._env.env._get_image()), win=self._win1)
            state = next_state.copy()
            sum_rwd += reward.numpy()
            if done:
                self._vis.line(X=np.array([n_episode]), Y=np.array([sum_rwd]),
                               win=self._win2, update='append')
                state = self._env.reset()
                sum_rwd = 0
                n_episode += 1
                step_buffer.clear()
            if len(self._local_memory) > self._batch_size:
                samples = self._local_memory.sample(self._batch_size)
                _, prio = self._policy_net.calc_priorities(self._policy_net, samples,
                                                           gamma=gamma_nsteps[-1],
                                                           device=self._device)
                print("[%s] Publish experience." % self._name)
                self._connect.rpush('experience',
                                    utils.dumps((samples, prio.squeeze(1).cpu().numpy().tolist())))
                self._local_memory.clear()

            if t % self._target_update == 0:
                self._pull_params()
