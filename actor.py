# -*- coding: utf-8 -*-
import sys
import gym
import numpy as np
from itertools import count
if sys.version_info.major == 3:
    import _pickle as cPickle
else:
    import cPickle
import redis
import torch
import visdom
from libs import replay_memory, utils, wrapped_env, models
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

class Actor(object):
    EPS_START = 1.0
    EPS_END = 0.1
    def __init__(self, name, hostname='localhost', batch_size=32, target_update=50, eps_decay=50000):
        self._env = gym.make('MultiFrameBreakout-v0')
        self._name = name
        self._batch_size = batch_size
        self._target_update = target_update
        self._eps_decay = eps_decay
        self._policy_net = models.DuelingDQN(self._env.action_space.n).to(device)
        self._target_net = models.DuelingDQN(self._env.action_space.n).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._win1 = vis.image(utils.preprocess(self._env.env._get_image()))
        self._win2 = vis.line(X=np.array([0]), Y=np.array([0.0]),
                              opts=dict(title='Score %s' % self._name))
        self._local_memory = replay_memory.ReplayMemory(1000)
        self._connect = redis.StrictRedis(host=hostname)

    def run(self):
        state = self._env.reset()
        sum_rwd = 0
        n_episode = 0
        for t in count():
            # Select and perform an action
            eps = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1. * t / self._eps_decay)
            action = utils.epsilon_greedy(torch.from_numpy(state).unsqueeze(0).to(device),
                                          self._policy_net, eps)
            next_state, reward, done, _ = self._env.step(action.item())
            reward = torch.tensor([reward])
            done = torch.tensor([float(done)])
            self._local_memory.push(torch.from_numpy(state), action,
                                    torch.from_numpy(next_state), reward, done)
            vis.image(utils.preprocess(self._env.env._get_image()), win=self._win1)
            state = next_state.copy()
            sum_rwd += reward.numpy()
            if done:
                state = self._env.reset()
                sum_rwd = 0
                n_episode += 1
                vis.line(X=np.array([n_episode]), Y=np.array([sum_rwd]), win=self._win2, update='append')
            if len(self._local_memory) > self._batch_size:
                samples = self._local_memory.sample(self._batch_size)
                _, prio = self._policy_net.calc_priorities(self._target_net, samples,
                                                           detach=True, device=device)
                print("[%s] Publish experience." % self._name)
                self._connect.rpush('experience',
                                    cPickle.dumps((samples, prio.squeeze(1).detach().cpu().numpy().tolist())))
                self._local_memory.clear()

            if t % self._target_update == 0:
                params = self._connect.get('params')
                if not params is None:
                    print("[%s] Sync params." % self._name)
                    self._policy_net.load_state_dict(cPickle.loads(params))
                    self._target_net.load_state_dict(self._policy_net.state_dict())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Actor for distributed reinforcement.')
    parser.add_argument('-n', '--name', type=str, default='actor1', help='Actor name.')
    args = parser.parse_args()
    actor = Actor(args.name)
    actor.run()
