# distributed_rl

This is pytorch implementation of distributed deep reinforcement learning.

* [ape-x](https://arxiv.org/abs/1803.00933)

## Install

```
sudo pip install pipenv
git clone https://github.com/neka-nat/distributed_rl.git
cd distributed_rl
pipenv install
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

Restart redis-server

```
sudo /etc/init.d/redis-server restart
```

## Run

```
pipenv shell
./run.sh <number of actors>
```

## System
In our system, there are two processes, Actor and Learner.
In Learner process, thread of the replay memory are running at the same time,
and these processes are communicating using Redis.

![system](images/system.png)

## Image

![image](images/image.gif)