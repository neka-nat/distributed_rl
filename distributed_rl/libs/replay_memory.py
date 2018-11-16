import random
from collections import deque
import numpy as np
from . import utils

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

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
    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = deque()
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
