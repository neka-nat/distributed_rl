import random
from collections import deque
import numpy as np
from libs import utils

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = utils.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.transitions = []
        self.priorities = []
        self.total_probs = 0.0
    
    def push(self, transitions, priorities):
        self.transitions.extend(transitions)
        self.priorities.extend(priorities)
        self.total_probs += sum(priorities)
        
    def sample(self, batch_size):
        batch = []
        idxs = []
        seg = self.total_probs / n

        idx = -1
        sum_p = 0
        for i in range(batch_size):
            s = random.uniform(seg * i, seg * (i + 1))
            while sum_p < s:
                sum_p += self.priorities[idx]
                idx += 1
            idxs.append(idx)
            batch.append(self.transition[idx])
        return batch, idxs
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.total_probs += (prio - self.priorities[idx])
            self.priorities[idx] = prio

    def remove_to_fit(self):
         for _ in range(len(self.priorities) - self.capacity):
            self.transition.popleft()
            p = self.priorities.popleft()
            self.total_probs -= p

    def __len__(self):
        return len(self.transitions)
