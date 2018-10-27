# -*- coding: utf-8 -*-
import numpy as np
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F
import visdom
from libs import utils, models
import replay
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

class Learner(object):
    def __init__(self, hostname='localhost'):
        self._policy_net = models.DuelingDQN(env.action_space.n).to(device)
        self._target_net = models.DuelingDQN(env.action_space.n).to(device)
        self._target_net.load_state_dict(policy_net.state_dict())
        self._target_net.eval()
        self._connect = redis.StrictRedis(host=hostname)
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), lr=0.00025 / 4, alpha=0.95, eps=0.01)
        self._memory = replay.Replay(50000, self._connect)
        self._memory.run()

    def optimize_loop(batch_size, beta=0.4, fit_timing=50):
        for t in count():
            if len(memory) < batch_size:
                return
            transitions, indices = memory.sample(batch_size)
            delta, prio = self._policy_net.calc_priorities(self._target_net,
                                                           transitions, device=device)
            total = len(self._memory)
            weights = (total * prio) ** (-beta)
            weights /= weights.max()
            weights = weights.to(device)
            loss = (delta * weights.unsqueeze(1)).mean()

            # Optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            for param in self._policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self._memory.update_priorities(indices, prio.data.cpu().numpy())
            self._optimizer.step()

if __name__ == '__main__':
    learner = Learner()
    learner.optimize_loop()
