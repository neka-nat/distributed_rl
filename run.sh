#!/bin/bash

trap 'kill $(jobs -p)' SIGINT SIGTERM EXIT

pids=""
python -m visdom.server -logging_level WARNING &
pids="$pids $!"

sleep 1

for i in `seq $1`
do
    python actor.py -n actor_$i &
    pids="$pids $!"
done

python learner.py &
pids="$pids $!"

wait -n $pids
