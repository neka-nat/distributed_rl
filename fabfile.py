#coding: utf-8
from fabric.api import run, cd
from fabric.api import env

def all_run():
    with cd('distributed_rl'):
        run("./run.sh 4")

def actor_run(num_proc='1', learner_host='localhost'):
    print('num_proc: %s' % num_proc)
    print('learner_host: %s' % learner_host)
    with cd('distributed_rl'):
        run("./run.sh %s config/actor.conf %s" % (num_proc, learner_host))

def learner_run():
    with cd('distributed_rl'):
        run("./run.sh 0 config/learner.conf localhost")
