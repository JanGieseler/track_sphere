import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from scipy.ndimage import measurements

from time import sleep
import sys, json

from track_sphere.utils import *
import matplotlib.pyplot as plt

def test_grab_frame(file_in, verbose=False):
    """

    tests if video can be opened with opencv and frames can be read

    Args:
        file_in:

    Returns:

    """
    cap = cv.VideoCapture(file_in, False)  # open input video

    ret, frame_in = cap.read()

    # show output
    # cv.imshow('frame', frame_in)

    cap.release()
    # cv.destroyAllWindows()

    if verbose:
        print(file_in, ':', ret)

    return ret

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

def features_surf(image, parameters, features = None, return_image=False):
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

    print('====>sadasdad', len(kp))

    print([k.pt for k in kp])

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
    # generate image that shows the detected features
    if return_image:
        # bright spots on magnet
        for k in range(min([len(kp), parameters['num_features']])):
            # print(k.angle, k.size, k.pt, (int(k.pt[0]), int(k.pt[1])))
            cv.circle(image, (int(kp[k].pt[0]), int(kp[k].pt[1])), 5, color=100)

        if features is not None:
            for lrc in lrcs:
                print('fffff', lrc)
                # top-left corner and bottom-right corner of rectangle.
                cv.rectangle(image, (lrc[0], lrc[1]), (lrc[0]+w, lrc[1]+h), (0, 255, 0), 1)

    else:
        image = None

    return data, image

def fit_blobs(image, parameters, points, return_image=False):
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
    plot_additions = []  # store the information to add the roi, ellipses to the image
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

        # threshold value
        data += [retVal]
        # ellipse center, size and angle
        data += list(ellipse[0]) + list(ellipse[1]) + [ellipse[2]]

        # save values for plotting
        if return_image:
            plot_additions.append([contour_magnet, ellipse, (c, r)])


    # generate image that shows the detected features
    if return_image:

        for i, (contour_magnet, ellipse, (c, r)) in enumerate(plot_additions):
            # outline of magnet contour
            cv.drawContours(image, [contour_magnet], -1, (0, 255, 0), 1)
            # initial point
            cv.circle(image, (c+int(0.5*w), r+int(0.5*h)), 2, (0, 0, 255), -1)
            # roi
            cv.rectangle(image, (c, r), (c+w, r+h), (255, 0, 0), 1)
            # outline of ellipse
            cv.ellipse(image, ellipse, (0, 0, 255), 1)
    else:
        image = None

    return data, image

def fit_ellipse(image, parameters, return_image=False):
    """
    fit an ellipse and tracks feature in image
    Args:
        image: image to be analysed
        return_image: if True returns image where showing all the features
        parameters: dictionary containing
        threshold, maxval, blockSize, c

    Returns:
        data, image with annotation

    """

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # ==  fit ellipse =====

    # mean threshold
    if parameters['threshold'] == 'mean':
        thresh = cv.adaptiveThreshold(gray, parameters['maxval'], cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, parameters['blockSize'], parameters['c'])
    elif parameters['threshold'] == 'gaussian':
        # gaussian threshold
        thresh = cv.adaptiveThreshold(gray, parameters['maxval'], cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, parameters['blockSize'], parameters['c'])
    elif isinstance(parameters['threshold'], int):
        # global threshold
        thresh = cv.threshold(gray, parameters['threshold'], parameters['maxval'], cv.THRESH_BINARY)[1]


    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_magnet = max(contours, key=len)

    # make sure it is convex
    if parameters['convex_hull']:
        contour_magnet = cv.convexHull(contour_magnet, returnPoints=True)


    M = cv.moments(contour_magnet)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    ellipse = cv.fitEllipse(contour_magnet)

    # contour center
    data = [cX, cY]

    # ellipse center, size and angle
    data += list(ellipse[0]) + list(ellipse[1]) + [ellipse[2]]

    # generate image that shows the detected features
    if return_image:
        # outline of magnet contour
        cv.drawContours(image, [contour_magnet], -1, (0, 255, 0), 1)
        # center of ellipse
        cv.circle(image, (cX, cY), 7, (0, 0, 255), -1)
        # outline of ellipse
        cv.ellipse(image, ellipse, (0, 0, 255), 1)
    else:
        image = None

    return data, image

def extract_position_data(file_in, file_out=None, min_frame = 0, max_frame = None, buffer_time=1e-6,
                          verbose = False, method='', method_parameters = None, export_parameters = {}):
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

    Returns:

    """



    ################################################################################
    #### checks the validity of the inputs and checks for existing files
    ################################################################################

    assert isinstance(file_in, str)

    if file_out is None:
        file_out = file_in.replace('.avi', '-{:s}.avi'.format(method))

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
            print('output file exists. Abort operation')
            return None

    if output_images>0:
        img_dir = file_out.replace('.avi', '-img')

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    #get the metadata from the input file
    info = load_video_info(file_in)


    if verbose:
        print(info)
    # if not specified, get code from input file
    if fourcc is None:
        fourcc = info['CodecID']

    assert isinstance(fourcc, str)
    assert len(fourcc) == 4

    if max_frame is None:
        max_frame = info['FrameCount']

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

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



    ################################################################################
    #### method dependent settings
    ################################################################################

    if method == 'BackgroundSubtractorMOG2':
        # fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False, history=5000) # create background substractor
        fgbg = cv.createBackgroundSubtractorMOG2()  # create background subtractor

        # the names of the data we will extract
        data_header = ['com x', 'com y', 'mean']
    elif method == 'grabCut':

        assert 'roi' in method_parameters
        assert 'iterations' in method_parameters

        mask = np.zeros((info['Width'], info['Height']), np.uint8)

        # Temporary array for the background/foreground model.
        # Not much info in the doc, e.g. don't know what a good length is.
        # For now just using the length from the tutorial (65)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # the names of the data we will extract
        data_header = ['com x', 'com y', 'mean']
    elif method == 'fit_ellipse':

        data_header = ['contour center x', 'contour center y']
        data_header += ['ellipse center x', 'ellipse center y', 'ellipse a', 'ellipse b', 'ellipse angle']
        # check and update the method_parameters dictionary
        if method_parameters is None:
            method_parameters = {}
        if 'threshold' not in method_parameters:
            # method_parameters['threshold'] = 100
            method_parameters['threshold'] = 'gaussian'
        if 'maxval' not in method_parameters:
            method_parameters['maxval'] = 255

        if 'convex_hull' not in method_parameters:
            method_parameters['convex_hull'] = True
        if method_parameters['threshold'] in ('mean', 'gaussian'):
            # Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
            if 'blockSize' not in method_parameters:
                method_parameters['blockSize'] = 21
            # Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
            if 'c' not in method_parameters:
                method_parameters['c'] = 2

    elif method == 'features_surf':
        data_header = sum([['k{:d} x'.format(i),
                            'k{:d} y'.format(i),
                            'k{:d} size'.format(i),
                            'k{:d} angle'.format(i)]
                           for i in range(method_parameters['num_features'])],[])
        if method_parameters is None:
            method_parameters = {}
        if 'xfeatures' not in method_parameters:
            method_parameters['xfeatures'] = 100
        if 'HessianThreshold' not in method_parameters:
            method_parameters['HessianThreshold'] = 1000
        if 'num_features' not in method_parameters:
            method_parameters['num_features'] = 5

    elif method == 'optical_flow':
        method_parameters['num_features'] = 5

    elif method == 'fit_blobs':
        if 'maxval' not in method_parameters:
            method_parameters['maxval'] = 255
        if 'convex_hull' not in method_parameters:
            method_parameters['convex_hull'] = False
        # method_parameters['num_features'] = 5
        # method_parameters['detect_points'] = True

        # take the center of the image as default
        if 'initial_points' not in method_parameters:
            method_parameters['initial_points'] = [[int(0.5*info['Width']), int(0.5*info['Height'])]]
        if 'winSize' not in method_parameters:
            method_parameters['winSize'] = (30,30)

        method_parameters['num_features'] = len(method_parameters['initial_points'])

        # threshold value, ellipse center (x,y), size (x,y) and angle
        data_header = sum([[
            'b{:d} thresh'.format(i),
            'b{:d} x'.format(i),
            'b{:d} y'.format(i),
            'b{:d} a'.format(i),
            'b{:d} b'.format(i),
            'b{:d} angle'.format(i)]
            for i in range(method_parameters['num_features'])], [])

        # if method_parameters['detect_points'] is False:
        # make sure that initial points 2D array with 2 elements per column

        print(method_parameters['initial_points'])
        print(np.shape(method_parameters['initial_points']))
        assert len(np.shape(method_parameters['initial_points'])) == 2
        assert len(method_parameters['initial_points'][0]) == 2


        points = method_parameters['initial_points']


    elif method == 'Bright px':
        # the names of the data we will extract
        data_header = ['bright px x', 'bright px y']
    elif method == 'optical_flow':
        data_header = sum([['k{:d} x'.format(i),
                            'k{:d} y'.format(i),
                            'k{:d} size'.format(i),
                            'k{:d} angle'.format(i)]
                           for i in range(5)],[])
        # check and update the method_parameters dictionary
        if method_parameters is None:
            method_parameters = {}
        if 'winSize' not in method_parameters:
            method_parameters['winSize'] = (15, 15)
        if 'maxLevel' not in method_parameters:
            method_parameters['maxLevel'] = 2
        if 'criteria' not in method_parameters:
            method_parameters['criteria'] = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    else:
        print('unknown method. Abort')
        return None


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
        frame_data = [] # this leads keeps the data per frame
        try:
            ret, frame_in = cap.read()
        except Exception as e:
            ret = False
        if ret:

            return_image = export_video or (output_images > 0 and frame_idx % output_images == 0)

            if method == 'BackgroundSubtractorMOG2':
                frame_out = fgbg.apply(frame_in)

                center_of_mass = measurements.center_of_mass(frame_out)
                frame_data = [center_of_mass[0], center_of_mass[1]]
                frame_data += [np.mean(frame_out)]

            elif method == 'grabCut':
                cv.grabCut(frame_in, mask, method_parameters['roi'], bgdModel, fgdModel, method_parameters['iterations'], cv.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                frame_out = frame_in * mask2[:,:,np.newaxis]

                center_of_mass = measurements.center_of_mass(frame_out)
                frame_data = [center_of_mass[0], center_of_mass[1]]
                frame_data += [np.mean(frame_out)]

            elif method == 'Bright px':
                frame_out = frame_in
                (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(frame_out[:,:,0], None)
                frame_data = [maxLoc[1], maxLoc[0]]

            elif method == 'fit_ellipse':
                frame_data, frame_out = fit_ellipse(frame_in, return_image=return_image, parameters=method_parameters)

            elif method == 'features_surf':
                frame_data, frame_out = features_surf(frame_in, return_image=return_image, parameters=method_parameters)

            elif method == 'fit_blobs':
                frame_data, frame_out = fit_blobs(frame_in, parameters=method_parameters, points=points, return_image=return_image)
                # retrieve the points for the next iteration
                points = points_from_blobs(frame_data, num_blobs=method_parameters['num_features'])

            # show output
            # cv.imshow('frame', frame_out)

            if export_video:
                if buffer_time>0:
                    sleep(buffer_time)
                video_writer.write(frame_out)
            # if frame_idx == 1:
            #     break
        else:
            skipped_frames.append(frame_idx)
            continue # go to next iteration

        if output_images>0 and frame_idx%output_images==0:
            cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}.jpg'.format(frame_idx))), frame_out)
            cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}_initit.jpg'.format(frame_idx))), frame_in)

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
    if not method_parameters is None:
        info_dict['method_parameters'] = method_parameters
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
