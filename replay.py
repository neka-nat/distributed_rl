# -*- coding: utf-8 -*-
import time
import threading
import redis
from libs import utils, replay_memory

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
            pipe = self._connect.pipeline()
            pipe.lrange('experience', 0, -1)
            pipe.ltrim('experience', -1, 0)
            data = pipe.execute()[0]
            if not data is None and len(data) > 0:
                trans, prios = [], []
                for d in data:
                    t, p = utils.loads(d)
                    trans.extend(t)
                    prios.extend(p)
                with self._lock:
                    self._memory.push(trans, prios)
            time.sleep(0.01)

    def update_priorities(self, indices, priorities):
        with self._lock:
            self._memory.update_priorities(indices, priorities)

    def remove_to_fit(self):
        with self._lock:
            self._memory.remove_to_fit()

    def sample(self, batch_size):
        with self._lock:
            return self._memory.sample(batch_size)

    def __len__(self):
        return len(self._memory)
