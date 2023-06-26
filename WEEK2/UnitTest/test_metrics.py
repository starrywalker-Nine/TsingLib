import unittest
from utils import metrics

class TestMetricsFunctions(unittest.TestCase):

    def test_mse(self):
        predictions = [1.0, 2.0, 3.0]
        targets = [1.5, 2.5, 3.5]
        expected = (0.5 ** 2 + 0.5 ** 2 + 0.5 ** 2) / 3
        self.assertAlmostEqual(metrics.mse(predictions, targets), expected)

    def test_mae(self):
        predictions = [1.0, 2.0, 3.0]
        targets = [1.5, 2.5, 3.5]
        expected = (0.5 + 0.5 + 0.5) / 3
        self.assertAlmostEqual(metrics.mae(predictions, targets), expected)

    def test_rmse(self):
        predictions = [1.0, 2.0, 3.0]
        targets = [1.5, 2.5, 3.5]
        expected = ((0.5 ** 2 + 0.5 ** 2 + 0.5 ** 2) / 3) ** 0.5
        self.assertAlmostEqual(metrics.rmse(predictions, targets), expected)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            metrics.mse([1, 2], [1])
        with self.assertRaises(ValueError):
            metrics.mae([1, 2], [1])
        with self.assertRaises(ValueError):
            metrics.rmse([1, 2], [1，2])

if __name__ == "__main__":
    unittest.main()