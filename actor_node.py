# -*- coding: utf-8 -*-
import argparse
import gym
import torch
import visdom
from distributed_rl.ape_x.actor import Actor
from distributed_rl.libs import wrapped_env, models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description='Actor process for distributed reinforcement.')
    parser.add_argument('-n', '--name', type=str, default='actor1', help='Actor name.')
    parser.add_argument('-e', '--env', type=str, default='MultiFrameBreakout-v0', help='Environment name.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-v', '--visdomserver', type=str, default='localhost', help="Visdom's server name.")
    args = parser.parse_args()
    vis = visdom.Visdom(server='http://' + args.visdomserver)
    env = gym.make(args.env)
    actor = Actor(args.name, env, models.DuelingDQN(env.action_space.n).to(device),
                  vis, hostname=args.redisserver)
    actor.run()
   
if __name__ == '__main__':
    main()