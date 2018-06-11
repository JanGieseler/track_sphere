import unittest
from track_sphere.extract_data_opencv import moments_roi
from track_sphere.utils import grab_frame
import cv2 as cv
import os
import numpy as np

import matplotlib.pyplot as plt


class TestFitFunctions(unittest.TestCase):
    """

    tests if video can be opened with opencv and frames can be read

    Args:
        file_in:

    Returns:

    """

    def setUp(self):
        folder_in = '../example_data/'
        filename_in = '20171207_magnet.avi'

        self.file_in = os.path.join(folder_in, filename_in)

    def test_moments_roi(self):
        file_in = self.file_in

        method_parameters = {}
        method_parameters['winSize'] = (20, 20)
        method_parameters['initial_points'] = [[60, 109], [89, 66], [91, 108], [96, 142], [139, 113]]

        image = grab_frame(file_in, 0)

        x, image = moments_roi(image, parameters=method_parameters, points=method_parameters['initial_points'],
                               return_image=True, verbose=False)


        print(x)

        plt.imshow(image); plt.show()

        input('any key to exit')

if __name__ == '__main__':
    unittest.main()



# ## TEST moments_roi
# folder_in = '../example_data/'
# filename_in = '20171207_magnet.avi'
#
# file_in = os.path.join(folder_in, filename_in)
#
# image = grab_frame(file_in, 0)
#
# method_parameters = {}
# method_parameters['winSize'] = (16, 16)
# method_parameters['initial_points'] = [[35, 60], [56, 60], [57, 82], [55, 35], [80, 60]]
#
# x, image = moments_roi(image, parameters= method_parameters, points= method_parameters['initial_points'], return_image=True, verbose=False)
#
#
# print(x)
#
# plt.imshow(image); plt.show()
#
# input('any key to exit')



#
# ## TEST moments_roi
# folder_in = '../example_data/'
# filename_in = '20171207_magnet.avi'
#
# file_in = os.path.join(folder_in, filename_in)
#
# image = grab_frame(file_in, 0)
#
# method_parameters = {}
# method_parameters['winSize'] = (16, 16)
# method_parameters['initial_points'] = [[35, 60], [56, 60], [57, 82], [55, 35], [80, 60]]
#
# x, image = moments_roi(image, parameters= method_parameters, points= method_parameters['initial_points'], return_image=True, verbose=False)
#
#
# print(x)
#
# plt.imshow(image); plt.show()
#
# input('any key to exit')



