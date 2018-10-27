# -*- coding: utf-8 -*-
import sys
import threading
if sys.version_info.major == 3:
    import _pickle as cPickle
else:
    import cPickle
import redis
import numpy as np
from libs import replay_memory, utils

class Replay(threading.Thread):
    def __init__(self, size=50000, connect=redis.StrictRedis(host='localhost')):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self._memory = replay_memory.PrioritizedMemory(size)
        self._connect = connect
        self._connect.delete('experience')
        self._lock = threading.Lock()

    def run(self):
        while True:
            data = self._connect.lpop('experience')
            if not data is None:
                trans, prios = cPickle.loads(data)
                with self._lock:
                    self._memory.push(trans, prios)

    def update_priorities(self, indices, priorities):
        self._memory.update_priorities(indices, priorities)

    def remove_to_fit(self):
        with self._lock:
            self._memory.remove_to_fit()

    def sample(self, batch_size):
        return self._memory.sample(batch_size)

    def __len__(self):
        return len(self._memory)
