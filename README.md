# distributed_rl

This is pytorch implementation of distributed deep reinforcement learning.

## Install

```
pipenv install
pipenv shell
```

```
sudo apt-get install redis-server
```

Edit /etc/redis/redis.conf

```
- bind 127.0.0.1
+ bind 0.0.0.0
```

```
- stop-writes-on-bgsave-error yes
+ stop-writes-on-bgsave-error no
```

## Run

```
python learner.py
```

```
python actor.py
```