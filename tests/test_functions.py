import unittest
from track_sphere.extract_data_opencv import moments_roi
from track_sphere.utils import grab_frame
import cv2 as cv
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from track_sphere.extract_data_opencv import check_method_parameters, load_video_info, get_frame_data, get_method_objects, process_image, add_features_to_image


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
    # unittest.main()
    import os
    import cv2 as cv

    folder_in = '../example_data/'
    filename_in = '20171207_magnet.avi'

    # ======== Settings ========
    folder_in = '../raw_data/20180607_Sample6_bead_1/'
    filename_in = '20180611_Sample_6_Bead_7.avi'

    frame_id = 9000


    # method = 'fit_ellipse_with_bgs'
    parameters = {
        'pre-processing': {'process_method': 'grabCut'},
        'extraction_parameters': {'method': 'fit_ellipse'}
    }
    # parameters['pre-processing']['roi'] = (20, 20, 160, 160)
    # parameters['pre-processing']['roi'] = (20, 20, 200, 200)
    # parameters['pre-processing']['roi'] = (10, 10, 230, 230)

    file_in = os.path.join(folder_in, filename_in)

    frame_in = grab_frame(file_in, verbose=True, frame_id=frame_id)


    info = load_video_info(file_in)

    parameters = check_method_parameters(parameters, info, verbose=True)

    processing_parameters = parameters['pre-processing']
    extraction_parameters = parameters['extraction_parameters']
    # # get headers for the data
    # data_header = get_data_header(extraction_parameters)
    # create method dependent objects
    method_objects = get_method_objects(parameters)



    print('asa', parameters)
    method_objects = get_method_objects(parameters)

    frame_out, feature_list = process_image(frame_in, pre_process_parameters=processing_parameters, method_objects=method_objects,
                              verbose=True, return_features=True)

    frame_out_pre = deepcopy(frame_out)
    #convert to color
    frame_out_pre = cv.cvtColor(frame_out_pre, cv.COLOR_GRAY2BGR)
    add_features_to_image(frame_out_pre, feature_list, verbose=True)

    print('===========>> extract')
    data, fl = get_frame_data(frame_out, parameters=extraction_parameters, return_features=True, method_objects=None, verbose=True)
    feature_list += fl
    frame_out = cv.cvtColor(frame_out, cv.COLOR_GRAY2BGR)
    add_features_to_image(frame_out, feature_list, verbose=True)
    print('pppppp', len(feature_list))
    # print('pppppp', feature_list)




    cv.imshow("out", np.hstack([frame_in, frame_out_pre, frame_out]))

    # print(type(frame_in))
    # print('asadsda', np.shape(frame_in))
    # frame_data, frame_out = get_frame_data(frame_in, method, method_parameters=method_parameters, verbose=True,
    #                                        method_objects=method_objects, return_image=True)
    #
    # print('asadsda', np.shape(frame_in), np.min(frame_in),np.max(frame_in))
    # print('asadsda', np.shape(frame_out))
    # print(method_parameters)
    #
    # #
    # # show the images
    # cv.imshow("original", frame_in)
    #
    # print(np.shape(frame_in))
    # print(np.shape(frame_out))
    #
    # cv.imshow("Edges", np.hstack([frame_out+frame_in]))
    # # cv.imshow("Edges", np.hstack([frame_in]))
    cv.waitKey(0)
    #





        #
        #
        #
        #
        #
        #
        # gray = cv.cvtColor(frame_in, cv.COLOR_BGR2GRAY)
        #
        #
        # # plt.hist(frame_in.flatten(), bins=25)
        # # plt.show()
        # # x = cv.calcHist(frame_in,channels=1, mask=None, histSize=10)
        # # print(x)
        # # input('hit any key to continue...')
        #
        # blurred = cv.GaussianBlur(gray, (3, 3), 0)
        #
        # wide = cv.Canny(blurred, 10, 200)
        # tight = cv.Canny(blurred, 225, 250)
        #
        #
        #
        # # A lower value of sigma  indicates a tighter threshold, whereas a larger value of sigma  gives a wider threshold.
        # # In general, you will not have to change this sigma  value often. Simply select a single, default sigma  value and apply it to your entire dataset of images.
        # sigma = 0.10
        # # compute the median of the single channel pixel intensities
        # v = np.median(blurred)
        #
        # # apply automatic Canny edge detection using the computed median
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))
        # auto = cv.Canny(blurred, lower, upper)
        #
        #
        # # show the images
        # cv.imshow("Original", frame_in)
        # cv.imshow("Edges", np.hstack([wide, tight, auto]))
        # cv.waitKey(0)
        #
        #
        #
        #
        #
        #
        #
