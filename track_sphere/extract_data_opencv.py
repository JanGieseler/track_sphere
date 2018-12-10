import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from scipy.ndimage import measurements

from time import sleep
import sys, json
from copy import deepcopy
from track_sphere.utils import *
import matplotlib.pyplot as plt

from track_sphere.read_write import load_video_info, load_video_info_xml

# def optical_flow_features_surf(image_old, image, features, parameters):
#     """
#     tracks motion features from image_old to image
#
#     Args:
#         image_old: previous frame
#         image:  new frame
#         features: features in previous frame
#         parameters: paramters for optical flow calculations, dictionary with:
#             winSize: 2 tuple
#             maxLevel: int
#             criteria: 3 tupple
#
#     Returns: features in image
#
#     """
#     # calculate optical flow
#     features_new, st, err = cv.calcOpticalFlowPyrLK(image_old, image, features, None, **parameters)
#     # Select good points
#     good_new = features_new[st == 1]
#     good_old = features[st == 1]
#
#     return good_new
from collections import namedtuple

Feature = namedtuple('Feature', 'type data color')


def is_in_roi(pt, roi):
    """
    checks if pt is in region of interest (roi)

    Args:
        pt:
        roi:

    Returns:

    """
    n_roi = True
    x, y = pt
    xo, yo, w, h = roi

    if x < xo-0.5*w:
        n_roi = False
    if x > xo+0.5*w:
        n_roi = False

    if y < yo-0.5*h:
        n_roi = False
    if y > yo+0.5*h:
        n_roi = False


    return n_roi

def features_surf(image, parameters, features = None, return_features=False):
    """
    finds features in image using the SURF algorithm
    Args:
        image: image to be analysed
        return_image: if True returns image where showing all the features
        parameters: dictionary containing
            xfeatures, HessianThreshold, num_features, winSize
        points: points where to look for features, this is the lower right corner of a rectangle with size winSize

        winSize: size of window, when points is provided

    Returns:
        data, image with annotation

    """
    # ==  feature detection =====

    surf = cv.xfeatures2d.SURF_create(parameters['xfeatures'])

    surf.setHessianThreshold(parameters['HessianThreshold'])


    if features is None:
        # if there are no points determined find features in whole image
        kp, des = surf.detectAndCompute(image, None)



    else:
        w, h = parameters['winSize']
        kp = []

        lrcs = lrc_from_features(features, parameters['winSize'])
        for i, (x, y) in enumerate(lrcs):
            # create a mask to focus on roi
            mask = np.zeros(image.shape[:2]).astype(np.uint8)
            cv.circle(mask, (x, y), w, 1, thickness=-1)
            kpo, des = surf.detectAndCompute(image, mask)


            # fig = plt.figure()
            # plt.imshow(mask)
            # fig.savefig('i.jpg')
            #

            print(np.sum(image), np.sum(mask))
            print(i,'==', x, y, kpo[0].pt)
            kp.append(kpo[0])
            #
            # subimg = cv.SetImageRoi(image, cv.Rect(lrc[0], lrc[1], w, h))
            #
            # subimg = image[lrc[1]:lrc[1]+h:, lrc[0]:lrc[0]+w]
            # kpo, des = surf.detectAndCompute(subimg, None)
            # print('===>', lrc, kpo)
            # print(np.shape(subimg))
            #
            #
            # x = subimg
            # if len(kpo)>0:
            #
            #     kp.append(kpo[0])



    # w, h = parameters['winSize']
    # kp, des = surf.detectAndCompute(image, None)
    # lrcs = lrc_from_features(features, parameters['winSize'])

    data = []
    # we expect num_features bright spots
    for i in range(parameters['num_features']):
        if i < len(kp):
            data += [kp[i].pt[0], kp[i].pt[1], kp[i].size,kp[i].angle]
        else:
            data += [None, None, None, None]


    # image  =  mask
    feature_list = []
    # generate image that shows the detected features
    if return_features:
        # bright spots on magnet
        for k in range(min([len(kp), parameters['num_features']])):
            feature_list.append(Feature('point', (int(kp[k].pt[0]), int(kp[k].pt[1])), None))
            # cv.circle(image, (int(kp[k].pt[0]), int(kp[k].pt[1])), 5, color=100)

        if features is not None:
            for lrc in lrcs:
                # top-left corner and bottom-right corner of rectangle.
                feature_list.append(Feature('roi', (lrc[0], lrc[1], lrc[0]+w, lrc[1]+h, None)))
                # cv.rectangle(image, (lrc[0], lrc[1]), (lrc[0]+w, lrc[1]+h), (0, 255, 0), 1)

    return data, feature_list

def moments_roi(image, parameters, points, return_features=False, verbose=False):
    """

    returns the centroid of the regions of interest that are centered around points

    Note that this function is not that great to get the location of the center of the bright spot if there is a non-zero background!

    Args:
        image:
        parameters:
        points:
        return_image:
        verbose:

    Returns:

    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    data = []  # store the data
    plot_additions = []  # store the information to add the roi, ellipses to the image
    w, h = parameters['winSize']

    features_list = []  # store the information to add the roi, ellipses to the image

    for i, pt in enumerate(points):

        r, c = int(pt[1]) - int(0.5*h), int(pt[0]) - int(0.5*w)
        # select subimage of interest
        gray_roi = gray[r:r+h, c:c+w]

        moments = cv.moments(gray_roi)

        cx, cy = moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
        cx, cy = cx+c, cy+r # correct for offset

        data += [cx, cy]

        # save values for plotting
        if return_features:
            features_list += [Feature('point', (cx, cy), None)]


    return data, features_list

def fit_blobs(image, parameters, points, return_features=False, verbose=False):
    """
    fit an ellipse and tracks feature in image
    Args:
        image: image to be analysed
        return_image: if True returns image where showing all the features
        parameters: dictionary containing
        winSize, maxval, convex_hull
        points: points around which we look for the blob

    Returns:
        data, image with annotation

    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    data = []  # store the data
    features_list = []  # store the information to add the roi, ellipses to the image
    w, h = parameters['winSize']

    for i, pt in enumerate(points):

        r, c = int(pt[1]) - int(0.5*h), int(pt[0]) - int(0.5*w)
        # select subimage of interest
        gray_roi = gray[r:r+h, c:c+w]

        # threshold image to identify the blob using otsu's method for automatic thresholding
        retVal, thresh = cv.threshold(gray_roi, 0, parameters['maxval'], cv.THRESH_BINARY + cv.THRESH_OTSU)

        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(c, r))

        contour_magnet = max(contours, key=len)

        # make sure contour is convex
        if parameters['convex_hull']:
            contour_magnet = cv.convexHull(contour_magnet, returnPoints=True)

        ellipse = cv.fitEllipse(contour_magnet)

        if verbose:
            if not is_in_roi(ellipse[0], (pt[0], pt[1], w, h)):
                print('find bolb is failing, try to reduce winSize')

        # threshold value
        data += [retVal]
        # ellipse center, size and angle
        data += list(ellipse[0]) + list(ellipse[1]) + [ellipse[2]]

        # save values for plotting
        if return_features:
            features_list += [
                Feature('contour', contour_magnet, None),
                Feature('ellipse', ellipse, None),
                Feature('point', (c, r), None)]

            # plot_additions.append([contour_magnet, ellipse, (c, r)])


    # # generate image that shows the detected features
    # if return_image:
    #
    #     for i, (contour_magnet, ellipse, (c, r)) in enumerate(plot_additions):
    #         # outline of magnet contour
    #         cv.drawContours(image, [contour_magnet], -1, (0, 255, 0), 1)
    #         # initial point
    #         cv.circle(image, (c+int(0.5*w), r+int(0.5*h)), 2, (0, 0, 255), -1)
    #         # roi
    #         cv.rectangle(image, (c, r), (c+w, r+h), (255, 0, 0), 1)
    #         # outline of ellipse
    #         cv.ellipse(image, ellipse, (0, 0, 255), 1)
    # else:
    #     image = None

    return data, features_list

def fit_ellipse(image_gray, parameters, return_features=False, verbose=False):
    """
    fit an ellipse and tracks feature in image
    Args:
        image: image to be analysed
        return_features: if True returns  all the features so that they can be added to the image later
        parameters: dictionary containing
        threshold, maxval, blockSize, c

    Returns:
        data, image with annotation

    """

    # expect a gray scale image
    assert len(np.shape(image_gray)) == 2

    im2, contours, hierarchy = cv.findContours(image_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # keep only the contours without a parent, i.e. only outer contours
    contours = [c for h, c in zip(hierarchy, contours) if h[-1] == -1]

    if parameters['select_contour'] == 'longest':
        #  select the longest contour
        contour_magnet = max(contours, key=len)
    elif parameters['select_contour'] == 'all':
        # compbine all contours into a single one
        contour_magnet = np.vstack(contours)
    else:
        raise ValueError('unknown value of select_contour parameter')



    if verbose:
        print('area of contour', cv.contourArea(contour_magnet))

    # make sure it is convex
    if parameters['convex_hull']:
        contour_magnet_hull = cv.convexHull(contour_magnet, returnPoints=True)
        if verbose:
            print('area of contour (convex_hull)', cv.contourArea(contour_magnet))
    else:
        contour_magnet_hull = contour_magnet





    if verbose:
        print('contour length', len(contour_magnet_hull))

    if len(contour_magnet_hull)>=5:
        ellipse = cv.fitEllipse(contour_magnet_hull)
        if verbose:
            print('area of ellipse', np.prod(ellipse[1]) * np.pi / 4)
    else:
        ellipse = [(None, None), (None, None), None]

    if ellipse[0] is not (None, None):
        # ellipse center
        cX, cY = ellipse[0]
    else:
        # contour center
        M = cv.moments(contour_magnet_hull)
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]

    data = [cX, cY]


    # ellipse center, size and angle
    data += list(ellipse[0]) + list(ellipse[1]) + [ellipse[2]]


    # generate image that shows the detected features
    if return_features:

        # major axis
        # from center
        # maj_ax = (np.asarray(ellipse[0], dtype=int),
        #           np.asarray([
        #               ellipse[0][0] + 0.5*ellipse[1][0] * np.cos(np.radians(ellipse[2])),
        #               ellipse[0][1] + 0.5*ellipse[1][0] * np.sin(np.radians(ellipse[2]))
        #           ],dtype=int))
        # across
        maj_ax = (
            np.asarray([
                ellipse[0][0] + 0.5 * ellipse[1][0] * np.cos(np.radians(ellipse[2])),
                ellipse[0][1] + 0.5 * ellipse[1][0] * np.sin(np.radians(ellipse[2]))
            ], dtype=int),
            np.asarray([
                ellipse[0][0] + 0.5*ellipse[1][0] * np.cos(np.pi+np.radians(ellipse[2])),
                ellipse[0][1] + 0.5*ellipse[1][0] * np.sin(np.pi+np.radians(ellipse[2]))
            ],dtype=int)
        )

        features = [
            Feature('contour', contour_magnet, None),
            Feature('contour', contour_magnet_hull, None),
            Feature('point', (int(cX), int(cY)), None),
            Feature('line', maj_ax, None)]


        # for cont in contours:
        #     features += [Feature('contour', cont, None)]


        if not ellipse[2] is None:
            features += [Feature('ellipse', ellipse, None)]
    else:
        features = []

    return data, features

def check_method_parameters(parameters, info=None, verbose=False):
    """

    check the parameter and set to default the parameters that are missing

    Args:
        parameters:
        info:
        verbose:

    Returns:

    """


    assert 'extraction_parameters' in parameters

    method = parameters['extraction_parameters']['method']
    if verbose:
        print('check_method_parameters: ', method)
    if not 'pre-processing' in parameters:
        parameters['pre-processing'] = {}


    ################################################################################
    #### pre-processing dependent settings
    ################################################################################
    if 'process_method' in parameters['pre-processing']:
        if parameters['pre-processing']['process_method'] == 'BackgroundSubtractorMOG2':
            if 'detectShadows' not in parameters['pre-processing']:
                parameters['pre-processing']['detectShadows'] = False
            if 'history' not in parameters['pre-processing']:
                parameters['pre-processing']['history'] = 5000

        elif parameters['pre-processing']['process_method'] == 'grabCut':
            if 'roi' not in parameters['pre-processing']:
                parameters['pre-processing']['roi'] = (20, 20, info['Width']-20, info['Height']-20)
            if 'mask_width' not in parameters['pre-processing']:
                parameters['pre-processing']['mask_width'] = info['Width']
            if 'mask_height' not in parameters['pre-processing']:
                parameters['pre-processing']['mask_height'] = info['Height']
            if 'iterations' not in parameters['pre-processing']:
                parameters['pre-processing']['iterations'] = 5

        elif parameters['pre-processing']['process_method'] in ['adaptive_thresh_mean', 'adaptive_thresh_gauss']:
            parameters['pre-processing']['maxval'] = 255
            parameters['pre-processing']['blockSize'] = 35
            parameters['pre-processing']['c'] = 11

        elif parameters['pre-processing']['process_method'] in ['threshold', 'thresh_triangle']:
            parameters['pre-processing']['maxval'] = 255

        elif parameters['pre-processing']['process_method'] == 'thresh_canny':
            parameters['pre-processing']['threshold_low'] = 50
            parameters['pre-processing']['threshold_high'] = 120

        elif parameters['pre-processing']['process_method'] == 'morph':
            if 'maxval' not in parameters['pre-processing']:
                parameters['pre-processing']['maxval'] = 255
            if 'blockSize' not in parameters['pre-processing']:
                parameters['pre-processing']['blockSize'] = 35
            if 'c' not in parameters['pre-processing']:
                parameters['pre-processing']['c'] = 11
            if 'k_size_noise' not in parameters['pre-processing']:
                parameters['pre-processing']['k_size_noise'] = 3
            if 'k_size_close' not in parameters['pre-processing']:
                parameters['pre-processing']['k_size_close'] = 11
            if 'threshold_type' not in parameters['pre-processing']:
                parameters['pre-processing']['threshold_type'] = 'mean'
            if 'normalize' not in parameters['pre-processing']:
                parameters['pre-processing']['normalize'] = True


        elif parameters['pre-processing']['process_method'] == 'roi':
            if 'roi' not in parameters['pre-processing']:
                parameters['pre-processing']['roi'] = (60, 60, 30, 30)

        elif parameters['pre-processing']['process_method'] == 'bilateral':
            if 'filter_dimension' not in parameters['pre-processing']:
                parameters['filter_dimension'] = 5
            if 'rescale' not in parameters['pre-processing']:
                parameters['normalize'] = True

            if 'sigmaColor' not in parameters['pre-processing']:
                parameters['sigmaColor'] = 50

            # if 'sigmaSpace' not in parameters['pre-processing']:
            #     parameters['sigmaSpace'] = 50

    else:
        parameters['pre-processing']['process_method'] = None


    ################################################################################
    #### method dependent settings
    ################################################################################
    if method == 'fit_ellipse':
        # if 'threshold' not in parameters['extraction_parameters']:
        #     # method_parameters['threshold'] = 100
        #     parameters['extraction_parameters']['threshold'] = 'gaussian'
        # if 'maxval' not in parameters['extraction_parameters']:
        #     parameters['extraction_parameters']['maxval'] = 255

        if 'convex_hull' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['convex_hull'] = True

        if 'select_contour' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['select_contour'] = 'longest'  # longest or all
        # if parameters['extraction_parameters']['threshold'] in ('mean', 'gaussian'):
        #     # Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
        #     if 'blockSize' not in parameters['extraction_parameters']:
        #         parameters['extraction_parameters']['blockSize'] = 21
        #     # Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
        #     if 'c' not in parameters['extraction_parameters']:
        #         parameters['extraction_parameters']['c'] = 2

    elif method == 'features_surf':
        if parameters['extraction_parameters'] is None:
            parameters = {}
        if 'xfeatures' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['xfeatures'] = 100
        if 'HessianThreshold' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['HessianThreshold'] = 1000
        if 'num_features' not in parameters:
            parameters['extraction_parameters']['num_features'] = 5

    elif method == 'fit_blobs':
        if 'maxval' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['maxval'] = 255
        if 'convex_hull' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['convex_hull'] = False


        # take the center of the image as default
        if 'initial_points' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['initial_points'] = [[int(0.5 * info['Width']), int(0.5 * info['Height'])]]
        if 'winSize' not in parameters:
            parameters['extraction_parameters']['winSize'] = (20, 20)
            parameters['extraction_parameters']['num_features'] = len(parameters['extraction_parameters']['initial_points'])

        assert len(np.shape(parameters['extraction_parameters']['initial_points'])) == 2
        assert len(parameters['extraction_parameters']['initial_points'][0]) == 2

    elif method == 'Bright px':
        pass
    elif method == 'optical_flow':
        # check and update the method_parameters dictionary
        if parameters['extraction_parameters'] is None:
            parameters['extraction_parameters'] = {}
        if 'winSize' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['winSize'] = (15, 15)
        if 'maxLevel' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['maxLevel'] = 2
        if 'criteria' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['criteria'] = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)

    elif method == 'moments_roi':

        # take the center of the image as default
        if 'initial_points' not in parameters['extraction_parameters']:
            parameters['extraction_parameters']['initial_points'] = [[int(0.5 * info['Width']), int(0.5 * info['Height'])]]
        if 'winSize' not in parameters:
            parameters['extraction_parameters']['winSize'] = (20, 20)
        if 'num_features' not in parameters:
            parameters['extraction_parameters']['num_features'] = len(parameters['extraction_parameters']['initial_points'])

        assert len(np.shape(parameters['extraction_parameters']['initial_points'])) == 2
        assert len(parameters['extraction_parameters']['initial_points'][0]) == 2
    else:
        print('unknown method. Abort', method)
        return None
    if verbose:
        print('===> check method parameters')

    return parameters

def get_data_header(method_parameters, verbose=False):
    ################################################################################
    #### method dependent settings
    ################################################################################
    method = method_parameters['method']

    if method == 'fit_ellipse':
        data_header = ['contour center x', 'contour center y']
        data_header += ['ellipse x', 'ellipse y', 'ellipse a', 'ellipse b', 'ellipse angle']
    elif method == 'features_surf':
        data_header = sum([['k{:d} x'.format(i),
                            'k{:d} y'.format(i),
                            'k{:d} size'.format(i),
                            'k{:d} angle'.format(i)]
                           for i in range(method_parameters['num_features'])], [])
    elif method == 'fit_blobs':
        # threshold value, ellipse center (x,y), size (x,y) and angle
        data_header = sum([[
            'b{:d} thresh'.format(i),
            'b{:d} x'.format(i),
            'b{:d} y'.format(i),
            'b{:d} a'.format(i),
            'b{:d} b'.format(i),
            'b{:d} angle'.format(i)]
            for i in range(method_parameters['num_features'])], [])
    elif method == 'Bright px':
        # the names of the data we will extract
        data_header = ['bright px x', 'bright px y']
    elif method == 'moments_roi':
        # the names of the data we will extract
        data_header = ['x', 'y']
    elif method == 'optical_flow':
        data_header = sum([['k{:d} x'.format(i),
                            'k{:d} y'.format(i),
                            'k{:d} size'.format(i),
                            'k{:d} angle'.format(i)]
                           for i in range(5)], [])
    else:
        print('unknown method. Abort')
        return None
    if verbose:
        print('===> setup methods')

    return data_header

def get_frame_data(frame, parameters, return_features=False, method_objects=None, verbose=False):


    # # if we return the image, make a deepcopy since the data stored in frame gets modified during processing
    # # the deepcopy makes sure that the data in frame_in remains untouched
    # if return_image:
    #     frame = deepcopy(frame_in)
    # else:
    #     frame = frame_in
    method = parameters['method']

    if verbose:
        print('method - get_frame_data', method)
    if method == 'Bright px':
        if len(np.shape(frame))==3:
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(frame[:, :, 0], None)
        else:
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(frame[:, :], None)
        frame_data = [maxLoc[1], maxLoc[0]]
        features = [Feature('point', (maxLoc[0], maxLoc[1]), None)]

    elif method == 'fit_ellipse':
        frame_data, features = fit_ellipse(frame, return_features=return_features, parameters=parameters, verbose=verbose)

    elif method == 'features_surf':
        frame_data, features = features_surf(frame, return_features=return_features, parameters=parameters)

    elif method == 'fit_blobs':
        points = method_objects['points']
        frame_data, features = fit_blobs(frame, parameters=parameters, points=points,
                                         return_features=return_features, verbose=verbose)
    elif method == 'moments_roi':
        points = method_objects['points']
        frame_data, features = moments_roi(frame, parameters=parameters, points=points, return_features=return_features, verbose=verbose)

    return frame_data, features

def process_image(frame, parameters, method_objects, verbose=False, return_features=False):
    """
    proccesses the frame, e.g. substract background, thresholding

    Args:
        frame:
        parameters:
        method_objects:

    Returns:

    """



    if verbose:
        print('process_method', parameters['process_method'])

    feature_list = []

    if parameters['process_method'] == 'BackgroundSubtractorMOG2':

        fgbg = method_objects['fgbg']
        frame_out = fgbg.apply(frame)


    elif parameters['process_method'] == 'grabCut':
        mask, bgdModel, fgdModel = method_objects['mask'], method_objects['bgdModel'], method_objects['fgdModel']
        cv.grabCut(frame, mask, parameters['roi'], bgdModel, fgdModel, parameters['iterations'],
                   cv.GC_INIT_WITH_RECT) #+cv.GC_INIT_WITH_MASK
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        frame_out = frame * mask2[:, :, np.newaxis]

        if return_features:
            # feature_list = []
            feature_list = [Feature('roi', parameters['roi'], None)]

        # convert to gray scale
        frame_out = cv.cvtColor(frame_out, cv.COLOR_BGR2GRAY)

    elif parameters['process_method'] in ['adaptive_thresh_mean', 'adaptive_thresh_gauss', 'threshold', 'thresh_canny', 'thresh_triangle']:
        # if we received a color image convert to gray scale
        if len(np.shape(frame)) == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif len(np.shape(frame)) == 2:
            gray = frame
        else:
            raise ValueError('unexpected shape')


        # mean threshold
        if parameters['process_method'] == 'adaptive_thresh_mean':
            maxval = parameters['maxval']
            blockSize = parameters['blockSize']
            c = parameters['c']
            frame_out = cv.adaptiveThreshold(gray, maxval, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, c)
        elif parameters['process_method'] == 'adaptive_thresh_gauss':
            # gaussian threshold
            maxval = parameters['maxval']
            blockSize = parameters['blockSize']
            c = parameters['c']
            frame_out = cv.adaptiveThreshold(gray, maxval, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, c)
        elif parameters['process_method'] == 'threshold':
            # global threshold
            maxval = parameters['maxval']
            threshold = parameters['threshold']
            frame_out = cv.threshold(gray, threshold, maxval, cv.THRESH_BINARY)[1]
        elif parameters['process_method'] == 'thresh_canny':
            threshold_low = parameters['threshold_low']
            threshold_high = parameters['threshold_high']
            frame_out = cv.Canny(gray, threshold1=threshold_low, threshold2=threshold_high)
        elif parameters['process_method'] == 'thresh_triangle':
            maxval = parameters['maxval']
            retVal, frame_out = cv.threshold(gray, 0, maxval, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

    elif parameters['process_method'] == 'morph':
        # if we received a color image convert to gray scale
        if len(np.shape(frame)) == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif len(np.shape(frame)) == 2:
            gray = frame
        else:
            raise ValueError('unexpected shape')

        maxval = parameters['maxval']
        blockSize = parameters['blockSize']
        c = parameters['c']

        k_size_noise = parameters['k_size_noise']
        k_size_close = parameters['k_size_close']
        threshold_type = parameters['threshold_type']

        if parameters['normalize']:
            # dst = np.zeros(shape=np.shape(frame_out))


            gray = cv.normalize(gray, None, 0, 255, norm_type=cv.NORM_MINMAX)

        if threshold_type == 'gauss':
            frame_out = cv.adaptiveThreshold(gray, maxval, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, c)
        elif threshold_type == 'mean':
            frame_out = cv.adaptiveThreshold(gray, maxval, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, c)
        else:
            raise TypeError


        frame_out = cv.bitwise_not(frame_out)


        # noise reduction
        if k_size_noise>0:
            kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size_noise, k_size_noise))
            frame_out = cv.morphologyEx(frame_out, cv.MORPH_OPEN, kernel_open, iterations=1)

        # close the contours
        if k_size_close > 0:
            kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size_close, k_size_close))
            frame_out = cv.morphologyEx(frame_out, cv.MORPH_CLOSE, kernel_close, iterations=1)

    elif parameters['process_method'] == 'roi':
        # print('adsdasda' , parameters['roi'])
        roi = parameters['roi']

        frame_out = np.zeros(frame.shape, np.uint8)
        frame_out[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]] = frame[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]]

    elif parameters['process_method'] == 'bilateral':
        filter_dimension = parameters['filter_dimension']
        sigmaColor = parameters['sigmaColor']

        # ksize = parameters['ksize']
        ksize = 3

        frame_out = cv.medianBlur(frame, ksize=ksize)

        if parameters['normalize']:
            # dst = np.zeros(shape=np.shape(frame_out))
            frame_out = cv.normalize(frame_out, None, 0, 255, norm_type=cv.NORM_MINMAX)

        frame_out = cv.bilateralFilter(frame_out, d=filter_dimension, borderType=cv.BORDER_ISOLATED, sigmaColor=sigmaColor, sigmaSpace=0)

        if 'roi' in parameters:
            roi = parameters['roi']

            frame_out_2 = np.zeros(frame_out.shape, np.uint8)
            frame_out_2[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]] = frame_out[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]]

            frame_out = frame_out_2


            # frame_out = dst

    elif parameters['process_method'] == None:
        frame_out = frame
    else:
        raise KeyError('did not find process_method')

    if verbose:
        print('cecksum frame_in', np.sum(frame))
        print('cecksum frame_out', np.sum(frame_out))

    return frame_out, feature_list

def get_method_objects(parameters):
    """
    returns the objets specific to a method
    Args:
        method:
        method_parameters:
        info:

    Returns:

    """
    pre_process_parameters = parameters['pre-processing']
    pre_process_method = pre_process_parameters['process_method']

    method_parameters = parameters['extraction_parameters']
    method = method_parameters['method']

    method_objects = {}

    ### objects for the preprocessing
    if pre_process_method == 'BackgroundSubtractorMOG2':
        # fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False, history=5000) # create background substractor
        fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=pre_process_parameters['detectShadows'], history=pre_process_parameters['history'])  # create background subtractor

        method_objects['fgbg'] = fgbg

    elif pre_process_method == 'grabCut':


        mask = np.zeros((pre_process_parameters['mask_width'], pre_process_parameters['mask_height']), np.uint8)
        # Temporary array for the background/foreground model.
        # Not much info in the doc, e.g. don't know what a good length is.
        # For now just using the length from the tutorial (65)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)


        method_objects.update({'mask': mask, 'bgdModel': bgdModel, 'fgdModel': fgdModel})

    ### objects for the interations
    if method in ['fit_blobs', 'moments_roi']:
        points = method_parameters['initial_points']
        method_objects['points'] = points


    return method_objects

def update_method_objects(parameters, method_objects, frame_data):
    """

    NOTE!! we might want to update method_objects instead of just overwritting

    updates the method objects based on the frame_data from the previous frame
    Args:
        method:
        parameters:
        method_objects:
        frame_data:

    Returns:



    """
    method = parameters['extraction_parameters']['method']
    if method == 'fit_blobs':
        # retrieve the points for the next iteration
        # method_objects.update({'points' : points_from_blobs(frame_data, num_blobs=parameters['extraction_parameters']['num_features'])
        method_objects['points'] = points_from_blobs(frame_data, num_blobs=parameters['extraction_parameters']['num_features'])
    return method_objects

def add_features_to_image(image, feature_list, verbose=False):

    for feature in feature_list:
        if feature.type == 'contour':
            # outline of contour
            # if verbose:
            #     print('adding contour')
            color = feature.color if feature.color is not None else (0, 255, 0)
            cv.drawContours(image, [feature.data], -1, color, 1)
        elif feature.type == 'point':
            # if verbose:
            #     print('adding point')
            cv.circle(image, (int(feature.data[0]), int(feature.data[1])), 3, (0, 0, 255), -1)
        elif feature.type == 'ellipse':
            # if verbose:
            #     print('adding ellipse')
            cv.ellipse(image, feature.data, (0, 0, 255), 1)
        elif feature.type == 'roi':
            # if verbose:
            #     print('adding roi')
            pt1 = tuple(feature.data[0:2])
            # pt2 = (pt1[0]+feature[2], pt1[1]+feature[3])
            pt2 = tuple(feature.data[2:4])
            cv.rectangle(image, pt1, pt2, (255, 0, 0), 1)
        elif feature.type == 'line':
            # if verbose:
            pt1 = tuple(feature.data[0])
            pt2 = tuple(feature.data[1])
            cv.line(image, pt1, pt2, (255, 0, 0), 2)

def extract_position_data(file_in, file_out=None, min_frame = 0, max_frame = None, buffer_time=1e-6,
                          verbose = False, parameters=None, stop_at_bad_frame=True):
    """
    Takes a video file and outputs a new file where the background is substracted
    Args:
        file_in:
        file_out:

        export_parameters: dictionary with the following parameters: export_video, output_fps, fourcc, output_images
            export_video: if True write output frames to video file
            output_fps: frames per second of output video file
            fourcc: four character code for video type (https://www.fourcc.org)
                if None use the same as the input video

                some options are
                // Lossless in quality with good processing performance
                'LAGS'
                // no compression. WARNING can take up a lot of space!!
                'FULL'

            output_images: (int) if specified output an image every output_images iterations

        buffer_time:  wait time between frames, needed because of some buffer issues of opencv

        method (str): 'BackgroundSubtractorMOG2', 'grabCut', 'Bright px', 'fit_ellipse', 'features_surf', 'optical_flow', 'fit_blobs'
        method_parameters: method specific parameters

        stop_at_bad_frame: if True the loop stops when a bad frame that can not be read is encountered. If false skip frame and go to next

    Returns:

    """


    assert 'extraction_parameters' in parameters
    # assert 'pre-processing' in parameters
    # assert 'export_parameters' in parameters

    method = parameters['extraction_parameters']['method']

    ################################################################################
    #### optional preprocessing steps that don't seem to be necessary, like reencoding with ffmepg and splitting the file into many small files
    ################################################################################

    if 'segmented' in parameters:
        #../raw_data/20180529_Sample6_bead_1_direct_thermal_01c-segmented/20180529_Sample6_bead_1_direct_thermal_01c-98.json
        segmented = parameters['segmented']
        # if the file has been segmented we actually take the video info from the original file
        file_in_info = os.path.join(os.path.dirname(os.path.dirname(file_in)), os.path.basename(os.path.dirname(file_in)).replace('-segmented', '.avi'))
    else:
        segmented = False
        file_in_info = file_in


    if 'reencode' in parameters:
        #../raw_data/20180529_Sample6_bead_1_direct_thermal_01c-segmented/20180529_Sample6_bead_1_direct_thermal_01c-98.json
        segmented = parameters['reencode']
        # if the file has been segmented we actually take the video info from the original file
        file_in_info = file_in
        file_in = file_in.replace('.avi', '-reencode.avi')
    else:
        reencode = False
        file_in_info = file_in



    # check what kind of info file we have json (ueye Camera) or xml (Phantom camera)
    if len(file_in_info.split('.avi')) == 2:
        file_in_info = file_in_info.replace('.avi', '.json')

        # if the info file is not a json file it is an xml file
        if not os.path.exists(file_in_info):
            file_in_info = file_in_info.replace('.json', '.xml')

    # check if it really exists
    assert os.path.exists(file_in_info)



    ################################################################################
    #### checks the validity of the inputs and checks for existing files
    ################################################################################
    if not 'export_parameters' in parameters:
        parameters['export_parameters'] = {}

    export_parameters = parameters['export_parameters']
    assert isinstance(file_in, str)

    if file_out is None:
        file_out = file_in_info.replace('.avi', '-{:s}.avi'.format(method))

    # set default values if not in dictionary
    if not 'export_video' in export_parameters:
        export_parameters['export_video'] = False
    if not 'output_fps' in export_parameters:
        export_parameters['output_fps'] = None
    if not 'fourcc' in export_parameters:
        export_parameters['fourcc'] = None
    if not 'output_images' in export_parameters:
        export_parameters['output_images'] = 1000

    export_video = export_parameters['export_video']
    output_fps = export_parameters['output_fps']
    fourcc = export_parameters['fourcc']
    output_images = export_parameters['output_images']

    if export_video:
        if os.path.exists(file_out):
            res = input('output file exists. Continue operation (y/n)')
            if not res == 'y':
                print('User stopped script')
                return None

    if output_images>0:
        img_dir = file_out.replace('.avi', '-img')

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    #get the metadata from the input file

    if len(file_in_info.split('.json')) == 2:
        info = load_video_info(file_in_info)
    elif len(file_in_info.split('.xml')) == 2:
        info = load_video_info_xml(file_in_info)
    else:
        raise ValueError('unknown info file format')

    if verbose:
        print('video info:')
        print(info)
    # if not specified, get code from input file
    if fourcc is None:
        fourcc = info['CodecID']

    assert isinstance(fourcc, str)
    assert len(fourcc) == 4

    if max_frame is None:
        max_frame = info['FrameCount']
    if verbose:
        print('===> passed validity test')

    ################################################################################
    #### setup input and output streams
    ################################################################################

    cap = cv.VideoCapture(file_in, False) #open input video
    cap.set(cv.CAP_PROP_POS_FRAMES, min_frame) # set the starting frame for reading to min frame

    if fourcc == 'FULL':
        # no compression
        fourcc = 0
    else:
        fourcc = cv.VideoWriter_fourcc(*fourcc)


    if export_video:

        if output_fps is None:
            output_fps = info['FrameRate']

        if os.name == 'posix':

            if method == 'BackgroundSubtractorMOG2':
                # last argument means that we load a black and white image
                video_writer = cv.VideoWriter(file_out, fourcc, output_fps, (info['Width'], info['Height']), False)
            elif method == 'grabCut':
                # for some reason there is an error when having the False argument and using grabcut
                video_writer = cv.VideoWriter(file_out, fourcc, output_fps, (info['Width'], info['Height']))
            else:
                video_writer = cv.VideoWriter(file_out, fourcc, output_fps, (info['Width'], info['Height']))
        else:
            # for windows doesn't work with False argument
            video_writer = cv.VideoWriter(file_out, fourcc, output_fps, (info['Width'], info['Height']))
    else:
        video_writer = None

    if verbose:
        print('===> set input output streams')

    ################################################################################
    #### check and setup parameters and create auxiliary objects
    ################################################################################

    # check the parameters
    parameters = check_method_parameters(parameters, info)

    processing_parameters = parameters['pre-processing']
    extraction_parameters = parameters['extraction_parameters']
    # get headers for the data
    data_header = get_data_header(extraction_parameters)
    # create method dependent objects
    method_objects = get_method_objects(parameters)

    ################################################################################
    #### start processing
    ################################################################################
    if not file_out is None:
        print('exporting video: {:s} => {:s}'.format(file_in,file_out))
    print('frames {:d}-{:d} ({:d})'.format(min_frame, max_frame, max_frame-min_frame))

    # the data set of the points we track
    data_set = []

    # keeps the skipped indecies
    skipped_frames = []

    sys.stdout.flush()
    for frame_idx in tqdm(range(min_frame, max_frame)):

        if verbose:
            print('frame id: ', frame_idx)

        try:
            ret, frame_in = cap.read()
        except Exception as e:
            ret = False

        if ret:

            # decide whether or not to return return_image_features
            return_image_features = export_video or (output_images > 0 and frame_idx % output_images == 0)

            if return_image_features:
                feature_list = []  # this keeps feature_list

            # preprocess image, e.g. thresholding, background subtraction
            frame_in_processed, fl = process_image(frame_in,
                                                   parameters=processing_parameters,
                                                   method_objects=method_objects,
                                                   return_features=return_image_features
                                                   )
            if return_image_features:
                feature_list += [fl]
            # extract the parameters from the preprocessed image
            frame_data, fl = get_frame_data(frame_in_processed,
                                            parameters=extraction_parameters,
                                            verbose=verbose,
                                            method_objects=method_objects,
                                            return_features=return_image_features)
            if return_image_features:
                feature_list += [fl]

            # update objects that change from one iteration to the next, e.g. center points of rois
            method_objects = update_method_objects(parameters, method_objects, frame_data)

            # add features from fit to image
            if return_image_features:
                frame_out = deepcopy(frame_in)

                add_features_to_image(frame_out, feature_list[1])

            if export_video:
                if buffer_time>0:
                    sleep(buffer_time)

                video_writer.write(frame_out)

        else:
            skipped_frames.append(frame_idx)

            if stop_at_bad_frame:
                break
            else:
                continue # go to next iteration


        if output_images>0 and frame_idx%output_images==0:
            #convert to color if gray scale image
            if len(np.shape(frame_in_processed))==2:
                frame_in_processed = cv.cvtColor(frame_in_processed, cv.COLOR_GRAY2BGR)
            # combine original image and the contours
            add_features_to_image(frame_in_processed, feature_list[0])
            # frame_out = cv.addWeighted(frame_in, 0.5, frame_out, 0.5, 0.0)

            # cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}.jpg'.format(frame_idx))), frame_out)


            cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}.jpg'.format(frame_idx))),
                       np.hstack([frame_in, frame_in_processed, frame_out]))

            # cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}_initit.jpg'.format(frame_idx))), frame_in)

        data_set.append(frame_data)

        k = cv.waitKey(30) & 0xff
        if k == 27:
            break


    ################################################################################
    #### clean up
    ################################################################################
    if export_video:
        video_writer.release()
    cap.release()
    cv.destroyAllWindows()


    ################################################################################
    #### export data
    ################################################################################

    data_dict = {k: v for k, v in zip(data_header, np.array(data_set).T)}

    df = pd.DataFrame.from_dict(data_dict)

    df.to_csv(file_out.replace('.avi','.dat'))

    #print meta data to json
    info_dict = {'info': info, 'method': method, 'skipped_frames': skipped_frames, 'export_parameters': export_parameters}
    if not parameters is None:
        info_dict['method_parameters'] = parameters
    with open(file_out.replace('.avi','.json'), 'w') as outfile:
        tmp = json.dump(info_dict, outfile, indent=4)


if __name__ == '__main__':
    # substract_background('test.avi', file_out=None)
    # info = load_video_info('test.avi')
    # print(info)
    from glob import glob
    import matplotlib.pyplot as plt

    filename ='magnet.jpg'

    filename = glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename))[0]

    img = cv.imread(filename)

    print(fit_ellipse(img, return_image=True))
    x = fit_ellipse(img, return_image=True)
    data, img = x
    plt.imshow(img), plt.show()


    # method = 'Bright px'
    # folder_in = './example_data/'
    # # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    # filename_in = '20171207_magnet.avi'
    #
    # folder_out = './data/'
    # filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))
    #
    # file_in = os.path.join(folder_in, filename_in)
    # file_out = os.path.join(folder_out, filename_out)
    #
    # print(file_in)
    # print(file_out)
    #
    #
    # substract_background(file_in, file_out=None, max_frame=None,output_images=0, verbose=True, method=method)
    #
