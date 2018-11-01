FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
RUN apt-get -y update && apt-get -y install redis-server
RUN pip install gym redis pillow atari-py visdom

RUN git clone https://github.com/neka-nat/distributed_rl.git
ENTRYPOINT /etc/init.d/redis-server start && /bin/bash