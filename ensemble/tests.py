import unittest
from simple_pred.function_set import *
from shared_tools.make_datasets import *
import simple_pred.pset


pset = simple_pred.pset.pset()

class TestStringMethods(unittest.TestCase):

    def test_extractors(self):
        for extractor in [all_sift, all_lbp, global_hog_small]:
            self.assertEqual(len(extractor(x_train[0]).shape), 1)

    def test_pset(self):
        image_processing_functions = [
            (laplace, 'Lap'), (gaussian_Laplace1, 'LoG1'),
            (gaussian_Laplace2, 'LoG2'), (sobelxy, 'Sobel'),
            (sobelx, 'SobelX'),(sobely, 'SobelY'), (medianf, 'Med'),
            (meanf, 'Mean'), (minf, 'Min'), (maxf, 'Max'), (lbp, 'LBP'),
            (hog_feature,'HoG'), (sqrt, 'Sqrt'), (relu, 'ReLU'),
        ]
        for img_processor, _ in image_processing_functions:
            self.assertNotEqual(len(img_processor(x_train[0]).shape), 0)


        print(gau(x_train[0], 1).shape)
        print(gauD(x_train[0], 1, 1, 1).shape)
        print(gab(x_train[0], 1, 1).shape)


    def extract_class(self, class_: type) -> list[type]:
        return pset.primitives[class_]
if __name__ == '__main__':
    unittest.main()
