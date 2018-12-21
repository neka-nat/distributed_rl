import random
from collections import deque
import numpy as np
from diskcache import Deque, Cache
from . import utils, sumtree

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
                      size_limit=int(10e9),
                      eviction_policy=u'least-frequently-used',
                      sqlite_journal_mode='memory')
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

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):
    def __init__(self, capacity,
                 use_compress=False,
                 use_disk=False):
        self.capacity = capacity
        self.transitions = generate_deque(use_compress, use_disk)
        self.priorities = sumtree.SumTree()
    
    def push(self, transitions, priorities):
        self.transitions.extend(transitions)
        self.priorities.extend(priorities)
        
    def sample(self, batch_size):
        idxs,  prios = self.priorities.prioritized_sample(batch_size)
        return [self.transitions[i] for i in idxs], prios, idxs
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def remove_to_fit(self):
        if len(self.priorities) - self.capacity <= 0:
            return
        for _ in range(len(self.priorities) - self.capacity):
            self.transitions.popleft()
            self.priorities.popleft()

    def __len__(self):
        return len(self.transitions)

    def total_prios(self):
        return self.priorities.root.value
