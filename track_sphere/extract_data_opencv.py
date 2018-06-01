import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import tqdm

from scipy.ndimage import measurements

import yaml, os


from time import sleep


from track_sphere.utils import *


def substract_background(file_in, file_out=None, min_frame = 0, max_frame = None, fourcc = None, output_images = 1000, buffer_time=1e-6):
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

    Returns:

    """



    ################################################################################
    #### checks the validity of the inputs and checks for existing files
    ################################################################################

    assert isinstance(file_in, str)


    if file_out is None:
        file_out = file_in.replace('.avi', '-no_bkgng.avi')


    if os.path.exists(file_out):
        print('output file exists. Abort operation')
        return None

    img_dir = file_out.replace('.avi', '-img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    #get the metadata from the input file
    info = load_video_info(file_in)

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
    # fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False, history=5000) # create background substractor
    fgbg = cv.createBackgroundSubtractorMOG2()  # create background substractor

    if fourcc == 'FULL':
        # no compression
        fourcc = 0
    else:
        fourcc = cv.VideoWriter_fourcc(*fourcc)

    # last argument means that we load a black and white image
    video_writer = cv.VideoWriter(file_out, fourcc, info['FrameRate'], (info['Width'], info['Height']), False)


    ################################################################################
    #### start processing
    ################################################################################
    print('subtracting background: {:s} => {:s}'.format(file_in,file_out))
    print('frames {:d}-{:d} ({:d})'.format(min_frame, max_frame, max_frame-min_frame))


    center_of_mass = []
    mean = []
    brightest_px = []

    for frame_idx in tqdm(range(max_frame)):

        ret, frame_in = cap.read()
        if ret:
            frame_out = fgbg.apply(frame_in)
            # cv.imshow('frame', frame_out)

            if buffer_time>0:
                sleep(buffer_time)
            video_writer.write(frame_out)

        if output_images>0 and frame_idx%output_images==0:
            # print('writing',frame_idx)

            cv.imwrite(os.path.join(img_dir,file_out.replace('.avi', '-{:d}.jpg'.format(frame_idx))), frame_out)
            cv.imwrite(os.path.join(img_dir, file_out.replace('.avi', '-{:d}_initit.jpg'.format(frame_idx))), frame_in)

        center_of_mass.append(measurements.center_of_mass(frame_out))
        mean.append(np.mean(frame_out))
        brightest_px.append(np.unravel_index(np.argmax(frame_in[:,:,0]), (info['Width'], info['Height'])))




        k = cv.waitKey(30) & 0xff
        if k == 27:
            break


    ################################################################################
    #### clean up
    ################################################################################
    video_writer.release()
    cap.release()
    cv.destroyAllWindows()

    center_of_mass = np.array([list(elem) for elem in center_of_mass])

    brightest_px = np.array(brightest_px)

    df = pd.DataFrame.from_dict({
        'center_of_mass x': center_of_mass[:,0],
        'center_of_mass y': center_of_mass[:,1],
        'bright x': brightest_px[:, 0],
        'bright y': brightest_px[:, 1],

        'mean': mean})

    df.to_csv(file_out.replace('.avi','.dat'))






if __name__ == '__main__':
    # substract_background('test.avi', file_out=None)
    # info = load_video_info('test.avi')
    # print(info)

    file_in = './raw_data/20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    file_out = './data/20180529_Sample6_bead_1_direct_thermal_01c_reencode-nobck.avi'
    substract_background(file_in, file_out=file_out, max_frame=5000,output_images=100)

