import random
from collections import deque
import numpy as np
from diskcache import Deque, Cache
from . import utils

def generate_deque(use_compress=False, use_disk=False, capacity=None):
    base_cls = deque if not use_disk else Deque
    if not use_compress:
        if capacity is None:
            return base_cls()
        else:
            return base_cls(maxlen=capacity)

    class CompressedDeque(base_cls):
        def __init__(self, *args, **kargs):
            super(CompressedDeque, self).__init__(*args, **kargs)

        def __iter__(self):
            return (utils.loads(v) for v in super(CompressedDeque, self).__iter__())

        def append(self, data):
            super(CompressedDeque, self).append(utils.dumps(data))

        def extend(self, datum):
            for d in datum:
                self.append(d)

        def __getitem__(self, idx):
            return utils.loads(super(CompressedDeque, self).__getitem__(idx))

    if use_disk:
        cache = Cache('/tmp/experience',
                      eviction_policy=u'least-frequently-used',
                      sqlite_mmap_size=int(4e9))
        cache.clear()
        return CompressedDeque.fromcache(cache)
    if capacity is None:
        return CompressedDeque()
    else:
        return CompressedDeque(maxlen=capacity)

class ReplayMemory(object):
    def __init__(self, capacity,
                 use_compress=False,
                 use_disk=False):
        if use_disk:
            self.memory = generate_deque(use_compress, use_disk)
        else:
            self.memory = generate_deque(use_compress, use_disk, capacity)

    def push(self, data):
        """Saves a transition."""
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):
    def __init__(self, capacity,
                 use_compress=False,
                 use_disk=False):
        self.capacity = capacity
        self.transitions = generate_deque(use_compress, use_disk)
        self.priorities = deque()
        self.total_prios = 0.0
    
    def push(self, transitions, priorities):
        self.transitions.extend(transitions)
        self.priorities.extend(priorities)
        self.total_prios += sum(priorities)
        
    def sample(self, batch_size):
        batch = []
        idxs = []
        seg = self.total_prios / batch_size

        idx = -1
        sum_p = 0
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            while sum_p < s:
                sum_p += self.priorities[idx]
                idx += 1
            idxs.append(idx)
            batch.append(self.transitions[idx])
        return batch, idxs
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.total_prios += (prio - self.priorities[idx])
            self.priorities[idx] = prio

    def remove_to_fit(self):
        if len(self.priorities) - self.capacity <= 0:
            return
        for _ in range(len(self.priorities) - self.capacity):
            self.transitions.popleft()
            p = self.priorities.popleft()
            self.total_prios -= p

    def __len__(self):
        return len(self.transitions)
