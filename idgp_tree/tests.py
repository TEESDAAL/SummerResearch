import unittest, complex_pred.function_set, complex_num_pred.function_set, MLGP.function_set, simple_pred.function_set, numpy as np


class TestStringMethods(unittest.TestCase):

    def test_selections(self):
        img = np.arange(100).reshape(10,10)
        [self.assertTrue(
            (np.array([[22,23],[32, 33]]) == fs.rect_region(img, 2,2,2,2)).all())
            for fs in (complex_pred.function_set, complex_num_pred.function_set, MLGP.function_set, simple_pred.function_set)]
    def test(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
