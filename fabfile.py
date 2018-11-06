#coding: utf-8
from fabric.api import run, cd
from fabric.api import env

def all_run():
    with cd('distributed_rl'):
        run("./run.sh 4")

def actor_run(num_proc=1, leaner_host='localhost'):
    with cd('distributed_rl'):
        run("./run.sh %d config/actor.conf %s" % (num_proc, leaner_host))

def learner_run():
    with cd('distributed_rl'):
        run("./run.sh 0 config/learner.conf localhost")
