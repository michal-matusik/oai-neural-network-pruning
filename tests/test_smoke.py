import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from pruning import global_magnitude_mask, apply_masks

class PruningSmokeTest(unittest.TestCase):
    def test_global_magnitude_pruning(self):
        params = [np.array([1., 2., 3.]), np.array([4.])]
        result = apply_masks(params, global_magnitude_mask(params, .5))
        self.assertEqual(sum(np.count_nonzero(x) for x in result), 2)
        self.assertTrue(np.array_equal(result[0], [0, 0, 3]))

if __name__ == '__main__': unittest.main()
