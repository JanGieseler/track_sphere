# this module contains a few functions that allow to test parameters on single frames before applying them to extract information from entire videos



from track_sphere.extract_data_opencv import fit_ellipse, features_surf, lrc_from_features, fit_blobs
import os
import cv2 as cv
from track_sphere.utils import select_initial_points
import matplotlib.pyplot as plt
import numpy as np


def check_parameters_fit_ellipse(file_in, method_parameters, frame = 0):
    """
    opens a video and runs fit_ellipse on the first frame using the parameters method_parameters
    shows the result
    Args:
        file_in:
        method_parameters:

    Returns: image

    """

    file_out = file_in.replace('.avi', 'test.jpg')

    cap = cv.VideoCapture(file_in)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)  # set the starting frame for reading to min frame

    # read frame
    ret, frame = cap.read()
    # fit ellipse
    data, frame_out = fit_ellipse(frame, method_parameters, return_image=True)



    print('fitellipse', data)

    cv.imwrite(file_out, frame_out)


    cap.release()
    cv.destroyAllWindows()

def check_parameters_fit_blobs(file_in, method_parameters, points, frame = 0):
    """
    opens a video and runs fit_blobson the first frame using the parameters method_parameters
    shows the result
    Args:
        file_in:
        method_parameters:

    Returns: image

    """

    file_out = file_in.replace('.avi', 'test.jpg')

    cap = cv.VideoCapture(file_in)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)  # set the starting frame for reading to min frame

    # read frame
    ret, frame = cap.read()
    # fit ellipse
    data, frame_out = fit_blobs(frame, method_parameters, points=points, return_image=True)



    print('fitblobs', data)

    cv.imwrite(file_out, frame_out)


    cap.release()
    cv.destroyAllWindows()

def check_parameters_features_surf(file_in, method_parameters, features=None, frame = 0):
    """
    opens a video and runs features_surf on the first frame using the parameters method_parameters
    shows the result
    Args:
        file_in:
        method_parameters:

    Returns: image

    """

    file_out = file_in.replace('.avi', 'test.jpg')

    cap = cv.VideoCapture(file_in)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)  # set the starting frame for reading to min frame

    # read frame
    ret, frame = cap.read()

    # if features is not None:
    #         for f in features:
    #     print

    # find features
    data, frame_out = features_surf(frame, method_parameters, features=features, return_image=True)
    print('data', data)



    # print('find features', lrc_from_features(features), method_parameters['winSize'])

    cv.imwrite(file_out, frame_out)


    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    case = 4

    # ========================================================
    # ==== Case 1 (fit_ellipse): the egg shaped magnet =======
    # ========================================================
    if case == 1:
        # ======== Settings ========
        method_parameters = {}
        method_parameters['threshold'] = 100
        method_parameters['maxval'] = 255

        # ======== Settings ========
        method_parameters = {}
        method_parameters['threshold'] = 'gaussian'
        # method_parameters['threshold'] = 'mean'
        # method_parameters['threshold'] = 100
        method_parameters['blockSize'] = 31
        method_parameters['c'] = 9
        method_parameters['maxval'] = 255
        method_parameters['convex_hull'] = True


        # give file to test
        folder_in = '../example_data/'
        # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
        filename_in = '20171207_magnet.avi'

        file_in = os.path.join(folder_in, filename_in)

        check_parameters_fit_ellipse(file_in, method_parameters, frame=0)

    # ==========================================================================
    # ==== Case 2 (fit_ellipse): a difficult one because of non-homogeneous background =======
    # ==========================================================================
    if case == 2:
        # ======== Settings ========
        method_parameters = {}
        method_parameters['xfeatures'] = 100
        method_parameters['HessianThreshold'] = 1000
        method_parameters['num_features'] = 5




        # give file to test
        folder_in = '../raw_data/'
        filename_in = '20180523_Sample6_bead_1_direct_thermal_03_reencode.avi'

        file_in = os.path.join(folder_in, filename_in)

        check_parameters_fit_ellipse(file_in, method_parameters, frame=1000)

    1768
    # ==========================================================================
    # ==== Case 3 (features_surf): the egg shaped magnet  ======================
    # ==========================================================================
    if case == 3:
        # ======== Settings ========
        method_parameters = {}
        method_parameters['xfeatures'] = 20
        method_parameters['HessianThreshold'] = 1000
        method_parameters['num_features'] = 5
        method_parameters['winSize'] = (30, 30)

        features = None
        # features = [[ 91.65412903, 102.88677979],
        #           [134.09725952, 105.9390564 ],
        #           [ 88.18359375,  55.8854332 ],
        #           [ 94.15112305, 136.26904297],
        #           [ 62.02745438, 101.42308044]
        #           ]
        features = [91.65412902832031, 102.88677978515625, 20.0, 144.75820922851562, 134.09725952148438, 105.93905639648438, 19.0, 122.29878234863281, 88.18359375, 55.885433197021484, 18.0, 148.00938415527344, 94.151123046875, 136.26904296875, 17.0, 211.0707550048828, 62.0274543762207, 101.42308044433594, 14.0, 122.03585052490234]

        print('find features', lrc_from_features(features, method_parameters['winSize']))
        # features = None

        # # give file to test
        folder_in = '../example_data/'
        filename_in = '20171207_magnet.avi'

        file_in = os.path.join(folder_in, filename_in)

        check_parameters_features_surf(file_in, method_parameters,features=features, frame=100)

    # ==========================================================================
    # ==== Case 4 (fit_blobs): the egg shaped magnet =======
    # ==========================================================================
    if case == 4:
        method = 'fit_blobs'
        # ======== Settings ========
        folder_in = '../example_data/'
        filename_in = '20171207_magnet.avi'

        export_video = False
        output_fps = 10
        output_images = 1000

        method_parameters = {}
        method_parameters['initial_points'] = [[60, 109], [89, 66], [91, 108], [96, 142], [139, 113]]
        method_parameters['winSize'] = (20,20)
        method_parameters['maxval'] = 255
        method_parameters['convex_hull'] = False


        # ----- end settings --------

        export_parameters = {
            'export_video': export_video,
            'output_fps': output_fps,
            'output_images': output_images
        }


        file_in = os.path.join(folder_in, filename_in)

        if method == 'fit_blobs' and method_parameters['initial_points'] is None:
            method_parameters['initial_points'] = select_initial_points(file_in)


        check_parameters_fit_blobs(file_in, method_parameters, points=method_parameters['initial_points'], frame=1775)
