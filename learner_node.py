# -*- coding: utf-8 -*-
import argparse
import gym
import torch
import visdom
from distributed_rl.ape_x.learner import Learner
from distributed_rl.libs import models, wrapped_env
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-e', '--env', type=str, default='MultiFrameBreakout-v0', help='Environment name.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-v', '--visdomserver', type=str, default='localhost', help="Visdom's server name.")
    parser.add_argument('-a', '--actordevice', type=str, default='', help="Actor's device.")
    parser.add_argument('-s', '--replaysize', type=int, default=30000, help="Replay memory size.")
    args = parser.parse_args()
    env = gym.make(args.env)
    vis = visdom.Visdom(server='http://' + args.visdomserver)
    actordevice = ("cuda" if torch.cuda.is_available() else "cpu") if args.actordevice == '' else args.actordevice
    learner = Learner(models.DuelingDQN(env.action_space.n).to(device),
                      models.DuelingDQN(env.action_space.n).to(device),
                      vis, replay_size=args.replaysize, hostname=args.redisserver)
    learner.optimize_loop(actor_device=torch.device(actordevice))

if __name__ == '__main__':
    main()
