#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT
config=${2:-"config/all.conf"}
source $config
if [ -z "$redis_server" ]; then
    redis_server=$3
fi

if [ -z "$visdom_server" ]; then
    visdom_server=$3
fi

echo "redis server:" $redis_server
echo "visdom server:" $visdom_server

pids=""

if $visdom; then
    python -m visdom.server -logging_level WARNING &
    pids="$pids $!"
    sleep 2
fi

if $actor; then
    for i in `seq $1`
    do
	python actor_node.py -n actor_$i -r $redis_server -v $visdom_server &
	pids="$pids $!"
    done
fi

if $leaner; then
    if [ -z "$actor_device" ]; then
	python learner_node.py -r $redis_server -v $visdom_server &
	pids="$pids $!"
    else
	python learner_node.py -r $redis_server -v $visdom_server -a $actor_device &
	pids="$pids $!"
    fi
fi

wait -n $pids
