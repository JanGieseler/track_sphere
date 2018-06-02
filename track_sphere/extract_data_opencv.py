import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from scipy.ndimage import measurements

import matplotlib.pyplot as plt


from time import sleep


from track_sphere.utils import *




def fit_ellipse(image, file_out=None, show_image=False):


    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    surf = cv.xfeatures2d.SURF_create(100)

    surf.setHessianThreshold(1000)

    kp, des = surf.detectAndCompute(image, None)


    thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)[1]
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contour_magnet = max(contours, key=len)

    M = cv.moments(contour_magnet)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    ellipse = cv.fitEllipse(contour_magnet)

    data_dict = {}
    # we expect 5 bright spots
    for i in range(5):
        if i<len(kp):
            data_dict.update({
                'k{:d} x'.format(i):kp[i].pt[0],
                'k{:d} y'.format(i):kp[i].pt[1],
                'k{:d} size'.format(i): kp[i].size,
                'k{:d} angle'.format(i): kp[i].angle
            })
        else:
            data_dict.update({
                'k{:d} x'.format(i): None,
                'k{:d} y'.format(i): None,
                'k{:d} size'.format(i): None,
                'k{:d} angle'.format(i): None
            })
    data_dict.update({
        'contour center x': cX,
        'contour center y': cY})

    data_dict = {k:[v] for k, v in data_dict.items()}


    data = pd.DataFrame.from_dict(data_dict)



    if show_image:

        # bright spots on magnet
        for k in kp:
            print(k.angle, k.size, k.pt, (int(k.pt[0]), int(k.pt[1])))
            cv.circle(image, (int(k.pt[0]), int(k.pt[1])), 5, color=100)
        # outline of magnet contour
        cv.drawContours(image, [contour_magnet], -1, (0, 255, 0), 1)
        # center of ellipse
        cv.circle(image, (cX, cY), 7, (0, 0, 255), -1)
        # outline of ellipse
        cv.ellipse(image, ellipse, (0, 0, 255), 1)

        plt.imshow(image), plt.show()

    return data, image


def substract_background(file_in, file_out=None, min_frame = 0, max_frame = None, fourcc = None, output_images = 1000, buffer_time=1e-6,
                         verbose = False, method='', method_parameters = None, export_video = False):
    """
    Takes a video file and outputs a new file where the background is substracted
    Args:
        file_in:
        file_out:


        fourcc: four character code for video type (https://www.fourcc.org)
            if None use the same as the input video

            some options are
            // Lossless in quality with good processing performance
            'LAGS'
            // no compression. WARNING can take up a lot of space!!
            'FULL'

        output_images: (int) if specified output an image every output_images iterations

        buffer_time:  wait time between frames, needed because of some buffer issues of opencv

        method (str): 'BackgroundSubtractorMOG2', 'grabCut', 'Bright px'
        method_parameters: method specific parameters

    Returns:

    """



    ################################################################################
    #### checks the validity of the inputs and checks for existing files
    ################################################################################

    assert isinstance(file_in, str)


    if file_out is None:
        file_out = file_in.replace('.avi', '-{:s}.avi'.format(method))

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

    ################################################################################
    #### setup input and output streams
    ################################################################################

    cap = cv.VideoCapture(file_in, False) #open input video

    if fourcc == 'FULL':
        # no compression
        fourcc = 0
    else:
        fourcc = cv.VideoWriter_fourcc(*fourcc)


    if export_video:
        if os.name == 'posix':
            if method == 'BackgroundSubtractorMOG2':
                # last argument means that we load a black and white image
                video_writer = cv.VideoWriter(file_out, fourcc, info['FrameRate'], (info['Width'], info['Height']), False)
            elif method == 'grabCut':
                # for some reason there is an error when having the False argument and using grabcut
                video_writer = cv.VideoWriter(file_out, fourcc, info['FrameRate'], (info['Width'], info['Height']))

        else:
            # for windows doesn't work with False argument
            video_writer = cv.VideoWriter(file_out, fourcc, info['FrameRate'], (info['Width'], info['Height']))
    else:
        video_writer = None

    ################################################################################
    #### method dependent settings
    ################################################################################

    if method == 'BackgroundSubtractorMOG2':
        # fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False, history=5000) # create background substractor
        fgbg = cv.createBackgroundSubtractorMOG2()  # create background substractor
    elif method == 'grabCut':

        assert 'roi' in method_parameters
        assert 'iterations' in method_parameters

        mask = np.zeros((info['Width'], info['Height']), np.uint8)

        # Temporary array for the background/foreground model.
        # Not much info in the doc, e.g. don't know what a good length is.
        # For now just using the length from the tutorial (65)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

    elif method == 'Bright px':
        pass
    else:


        print('unknown method. Abort')
        return None


    ################################################################################
    #### start processing
    ################################################################################
    if not file_out is None:
        print('subtracting background: {:s} => {:s}'.format(file_in,file_out))
    print('frames {:d}-{:d} ({:d})'.format(min_frame, max_frame, max_frame-min_frame))


    center_of_mass = []
    mean = []
    brightest_px = []

    for frame_idx in tqdm(range(max_frame)):

        ret, frame_in = cap.read()
        if ret:

            if method == 'BackgroundSubtractorMOG2':
                frame_out = fgbg.apply(frame_in)
            elif method == 'grabCut':
                cv.grabCut(frame_in, mask, method_parameters['roi'], bgdModel, fgdModel, method_parameters['iterations'], cv.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                frame_out = frame_in * mask2[:,:,np.newaxis]

            elif method == 'Bright px':
                pass


            # show output
            # cv.imshow('frame', frame_out)

            if export_video:
                if buffer_time>0:
                    sleep(buffer_time)
                video_writer.write(frame_out)
            # if frame_idx == 1:
            #     break

        if output_images>0 and frame_idx%output_images==0:
            cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}.jpg'.format(frame_idx))), frame_out)
            cv.imwrite(os.path.join(img_dir, os.path.basename(file_out).replace('.avi', '-{:d}_initit.jpg'.format(frame_idx))), frame_in)

        if method in ['BackgroundSubtractorMOG2', 'grabCut']:
            center_of_mass.append(measurements.center_of_mass(frame_out))
            mean.append(np.mean(frame_out))
        brightest_px.append(np.unravel_index(np.argmax(frame_in[:,:,0]), (info['Width'], info['Height'])))




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

    data_dict = {}
    if len(center_of_mass)>0:
        center_of_mass = np.array([list(elem) for elem in center_of_mass])
        data_dict.update({
            'center_of_mass x': center_of_mass[:,0],
            'center_of_mass y': center_of_mass[:,1]
        })

    if len(mean) > 0:
        data_dict.update({
            'mean': mean
        })

    brightest_px = np.array(brightest_px)

    data_dict.update({
        'bright x': brightest_px[:, 0],
        'bright y': brightest_px[:, 1],
    })

    df = pd.DataFrame.from_dict(data_dict)


    df.to_csv(file_out.replace('.avi','.dat'))
    print(file_out.replace('.avi','.dat'))




if __name__ == '__main__':
    # substract_background('test.avi', file_out=None)
    # info = load_video_info('test.avi')
    # print(info)
    from glob import glob


    filename ='magnet.jpg'

    filename = glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename))[0]

    img = cv.imread(filename)


    fit_ellipse(img, file_out='asda')


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
