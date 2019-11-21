#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT
config=${2:-"config/all.conf"}
source $config
algorithm=$algorithm
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

if $leaner; then
    if [ -z "$actor_device" ]; then
	python learner_node.py -r $redis_server -v $visdom_server -a $algorithm &
	pids="$pids $!"
    else
	python learner_node.py -r $redis_server -v $visdom_server -d $actor_device -a $algorithm &
	pids="$pids $!"
    fi
fi

if $actor; then
    for i in `seq $1`
    do
	python actor_node.py -n $i -t $1 -r $redis_server -v $visdom_server -a $algorithm &
	pids="$pids $!"
    done
fi

wait -n $pids
