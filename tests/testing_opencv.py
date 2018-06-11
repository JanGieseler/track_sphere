import unittest
# the goal of this file is to check if open cv works properly, i.e. if we can read and write files


import cv2 as cv
import os
import numpy as np

from track_sphere.utils import grab_frame




class TestGrabFrame(unittest.TestCase):
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

    def test_grab_frame(self):


        file_in = self.file_in
        frame_in = grab_frame(file_in, verbose=True)



if __name__ == '__main__':
    unittest.main()



# def test_grab_frame(file_in, verbose=False):
#     """
#
#     tests if video can be opened with opencv and frames can be read
#
#     Args:
#         file_in:
#
#     Returns:
#
#     """
#     cap = cv.VideoCapture(file_in, False)  # open input video
#
#     ret, frame_in = cap.read()
#
#     # show output
#     # cv.imshow('frame', frame_in)
#
#     cap.release()
#     # cv.destroyAllWindows()
#
#     if verbose:
#         print(file_in, ':', ret)
#
#     return ret


# folder_in = '../example_data/'
# filename_in = '20171207_magnet.avi'
#
# file_in = os.path.join(folder_in, filename_in)
#
# frame_id = 0
# file_out = file_in.replace('.avi', '-test-{:d}.jpg'.format(frame_id))
#
# print('START TEST')
# print('{:s} => {:s}'.format(file_in, file_out))
# cap = cv.VideoCapture(file_in)
# cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)  # set the starting frame for reading to min frame
# # read frame
# ret, image = cap.read()
#
# print('read frame', ret, np.shape(image))
# # convert to gray image
# frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
# cv.imwrite(file_out, image)
# print('wrote image to file: {:s}'.format(file_out))
#
# # input('anykey...')
# cap.release()
#
# cv.destroyAllWindows()
#
# print('FINISHED TEST')
