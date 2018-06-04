from track_sphere.extract_data_opencv import fit_ellipse
import os
import cv2 as cv
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
    method_parameters['detect_features'] = True

    file_out = file_in.replace('.avi', 'test.jpg')

    cap = cv.VideoCapture(file_in)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)  # set the starting frame for reading to min frame

    # read frame
    ret, frame = cap.read()
    # fit ellipse and track features
    data, frame_out = fit_ellipse(frame, method_parameters, return_image=True)
    print('fitellipse', data)

    cv.imwrite(file_out, frame_out)


    cap.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    # ======== Settings ========
    method_parameters = {}
    method_parameters['xfeatures'] = 100
    method_parameters['HessianThreshold'] = 1000
    method_parameters['threshold'] = 100
    method_parameters['maxval'] = 255
    method_parameters['num_features'] = 5

    # give file to test
    folder_in = '../example_data/'
    # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    filename_in = '20171207_magnet.avi'

    file_in = os.path.join(folder_in, filename_in)

    check_parameters_fit_ellipse(file_in, method_parameters, frame=1000)

    # fig = plt.fig
    # # show the image and the features
    # plt.imshow(np.array(image[:, :, 0]))
    #
    # input("Press Enter to continue...")


