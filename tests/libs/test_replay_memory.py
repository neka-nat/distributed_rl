import sys
sys.path.append("../..")
import unittest
import numpy as np
import torch
from distributed_rl.libs import replay_memory

class TestReplayMemory(unittest.TestCase):
    def test_compressed_deque(self):
        dq = replay_memory.CompressedDeque(maxlen=10)
        dq.append(1)
        self.assertEqual(len(dq), 1)
        dq.extend(range(9))
        self.assertEqual(len(dq), 10)
        dq.append(10)
        self.assertEqual(len(dq), 10)
        self.assertEqual(dq[-1], 10)

if __name__ == "__main__":
    unittest.main()
