#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT
config=${2:-"config/all.conf"}
source $config

echo "redis server:" $redis_server
echo "visdom server:" $visdom_server

pids=""

if $redis; then
    redis-server --bind 0.0.0.0 --stop-writes-on-bgsave-error no &
    pids="$pids $!"
    sleep 1
fi

if $visdom; then
    python -m visdom.server -logging_level WARNING &
    pids="$pids $!"
    sleep 1
fi

if $actor; then
    for i in `seq $1`
    do
	python actor.py -n actor_$i -r $redis_server -v $visdom_server &
	pids="$pids $!"
    done
fi

if $leaner; then
    python learner.py -r $redis_server -v $visdom_server &
    pids="$pids $!"
fi

wait -n $pids
