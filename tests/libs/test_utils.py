import sys
sys.path.append("../..")
import unittest
import numpy as np
import torch
from distributed_rl.libs import utils

class TestUtils(unittest.TestCase):
    def test_preprocess(self):
        src = np.random.randint(256, size=(10, 10, 3), dtype=np.uint8)
        dst = utils.preprocess(src)
        self.assertEqual(src.shape[:2], dst.shape[1:])
        self.assertEqual(src.shape[2], dst.shape[0])
        # resize image
        dst = utils.preprocess(src, shape=(5, 5))
        self.assertEqual(dst.shape[1:], (5, 5))
        # make gray-scale image
        dst = utils.preprocess(src, gray=True)
        self.assertEqual(dst.shape, src.shape[:2])

    def test_dumps_loads(self):
        src = torch.rand(10)
        dst = utils.loads(utils.dumps(src))
        torch.testing.assert_allclose(src, dst)

    def test_rescale(self):
        src = 10 * (torch.rand(10) * 2.0 - 1.0)
        dst1 = utils.inv_rescale(utils.rescale(src, 0.1), 0.1)
        dst2 = utils.rescale(utils.inv_rescale(src, 0.1), 0.1)
        torch.testing.assert_allclose(src, dst1)
        torch.testing.assert_allclose(src, dst2)
        dst1 = utils.inv_rescale(utils.rescale(src, 0), 0)
        dst2 = utils.rescale(utils.inv_rescale(src, 0), 0)
        torch.testing.assert_allclose(src, dst1)
        torch.testing.assert_allclose(src, dst2)

if __name__ == "__main__":
    unittest.main()
