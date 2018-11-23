# -*- coding: utf-8 -*-
import numpy as np
from itertools import count
from collections import deque
import redis
import torch
from ..ape_x import actor
from ..libs import replay_memory, utils
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(actor.Actor):
    EPS_START = 1.0
    EPS_END = 0.1
    def __init__(self, name, env, policy_net, target_net, vis, hostname='localhost',
                 batch_size=20, target_update=400, eps_decay=20000):
        super(Actor, self).__init__(name, env, policy_net, vis, hostname,
                                    batch_size, target_update, eps_decay)
        self._target_net = target_net
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

    def run(self, n_burn_in=40, n_sequence=80, nstep_return=5, gamma=0.997,
            clip=lambda x: x):
        assert n_burn_in < n_sequence, "n_burn_in must be less than n_sequence."
        state = self._env.reset()
        step_buffer = deque(maxlen=nstep_return)
        recurrent_state_buffer = deque(maxlen=2)
        sequence_buffer = []
        gamma_nsteps = [gamma ** i for i in range(nstep_return + 1)]
        recurrent_state_buffer.append(self._policy_net.get_state())
        sum_rwd = 0
        n_episode = 0
        for t in count():
            if len(sequence_buffer) == n_sequence - n_burn_in - step_buffer.maxlen:
                recurrent_state_buffer.append(self._policy_net.get_state())
            # Select and perform an action
            eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * t / self._eps_decay)
            action = utils.epsilon_greedy(torch.from_numpy(state).unsqueeze(0).to(device),
                                          self._policy_net, eps)
            next_state, reward, done, _ = self._env.step(action.item())
            reward = torch.tensor([clip(reward)])
            done = torch.tensor([float(done)])
            step_buffer.append(utils.Transition(torch.from_numpy(state), action, reward))
            if len(step_buffer) == step_buffer.maxlen:
                r_nstep = sum([gamma_nsteps[-(i + 2)] * step_buffer[i].reward for i in range(step_buffer.maxlen)])
                sequence_buffer.append(utils.Transition(step_buffer[0].state, step_buffer[0].action, r_nstep))
            if len(sequence_buffer) == n_sequence:
                self._local_memory.push(utils.Sequence(sequence_buffer, recurrent_state_buffer[0]))
                sequence_buffer = sequence_buffer[-n_burn_in:]
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
                sequence_buffer = []
                self._policy_net.reset(done)
                recurrent_state_buffer.append(self._policy_net.get_state())
            if len(self._local_memory) > self._batch_size:
                samples = self._local_memory.sample(self._batch_size)
                recurrent_state = self._policy_net.get_state()
                _, prio = self._policy_net.calc_priorities(self._target_net, samples,
                                                           gamma=gamma_nsteps[-1], device=device)
                self._policy_net.set_state(recurrent_state, device)
                print("[%s] Publish experience." % self._name)
                self._connect.rpush('experience',
                                    utils.dumps((samples, prio.squeeze(1).cpu().numpy().tolist())))
                self._local_memory.clear()

            if t % self._target_update == 0:
                self._pull_params()
                self._target_net.load_state_dict(self._policy_net.state_dict())
