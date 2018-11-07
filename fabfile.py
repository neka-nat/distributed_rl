#coding: utf-8
from fabric.api import run, cd, settings
from fabric.api import env

def all_run(num_proc='4'):
    with cd('distributed_rl'):
        with settings(warn_only=True):
            run("./run.sh %s" % num_proc)

def actor_run(num_proc='1', learner_host='localhost'):
    print('num_proc: %s' % num_proc)
    print('learner_host: %s' % learner_host)
    with cd('distributed_rl'):
        with settings(warn_only=True):
            run("./run.sh %s config/actor.conf %s" % (num_proc, learner_host))

def learner_run():
    with cd('distributed_rl'):
        with settings(warn_only=True):
            run("./run.sh 0 config/learner.conf localhost")
