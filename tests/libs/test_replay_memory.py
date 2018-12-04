import sys
sys.path.append("../..")
import unittest
import numpy as np
import torch
from distributed_rl.libs import replay_memory

class TestReplayMemory(unittest.TestCase):
    def test_compressed_deque(self):
        # in-memory
        dq = replay_memory.generate_deque(True, False, 10)
        dq.append(1)
        self.assertEqual(len(dq), 1)
        dq.extend(range(9))
        self.assertEqual(len(dq), 10)
        dq.append(10)
        self.assertEqual(len(dq), 10)
        self.assertEqual(dq[-1], 10)
        # use disk
        dq = replay_memory.generate_deque(True, True)
        dq.clear()
        dq.append(1)
        self.assertEqual(len(dq), 1)
        dq.extend(range(9))
        self.assertEqual(len(dq), 10)
        dq.append(10)
        self.assertEqual(len(dq), 11)
        self.assertEqual(dq[-1], 10)
        dq.clear()

if __name__ == "__main__":
    unittest.main()
