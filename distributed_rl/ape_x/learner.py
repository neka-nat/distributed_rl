# -*- coding: utf-8 -*-
import time
import numpy as np
from itertools import count
import redis
import torch
import torch.optim as optim
from ..libs import utils
from . import replay
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Learner(object):
    def __init__(self, policy_net, target_net,
                 vis, replay_size=30000, hostname='localhost',
                 lr=0.00025 / 4, alpha=0.95, eps=1.5e-7):
        self._vis = vis
        self._policy_net = policy_net
        self._target_net = target_net
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._connect = redis.StrictRedis(host=hostname)
        self._connect.delete('params')
        self._optimizer = optim.RMSprop(self._policy_net.parameters(), lr=lr, alpha=alpha, eps=eps)
        self._win = self._vis.line(X=np.array([0]), Y=np.array([0]),
                             opts=dict(title='Memory size'))
        self._memory = replay.Replay(replay_size, self._connect)
        self._memory.start()

    def optimize_loop(self, batch_size=512, nstep_return=3, gamma=0.999,
                      beta=0.4, fit_timing=100, target_update=1000, actor_device=device):
        gamma_nstep = gamma ** nstep_return
        for t in count():
            if len(self._memory) < batch_size:
                continue
            transitions, indices = self._memory.sample(batch_size)
            delta, prio = self._policy_net.calc_priorities(self._target_net,
                                                           transitions, gamma=gamma_nstep,
                                                           device=device)
            total = len(self._memory)
            weights = (total * prio.cpu().numpy()) ** (-beta)
            weights /= weights.max()
            loss = (delta * torch.from_numpy(np.expand_dims(weights, 1)).to(device)).mean()

            # Optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            for param in self._policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self._memory.update_priorities(indices,
                                           prio.squeeze(1).cpu().numpy().tolist())
            self._optimizer.step()

            self._connect.set('params', utils.dumps(self._policy_net.to(actor_device).state_dict()))
            self._policy_net.to(device)
            if t % fit_timing == 0:
                print('[Learner] Remove to fit.')
                self._memory.remove_to_fit()
                self._vis.line(X=np.array([t]), Y=np.array([len(self._memory)]),
                               win=self._win, update='append')
            if t % target_update == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())
            time.sleep(0.01)
