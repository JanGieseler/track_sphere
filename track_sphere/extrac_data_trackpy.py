import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.optimize as sopt
import glob

from tqdm import tqdm

import pims
from pims import pipeline
from skimage.color import rgb2gray
rgb2gray_pipeline = pipeline(rgb2gray)
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass

import trackpy as tp
# status bar

from IPython.display import display

from imageio.core import CannotReadFrameError


import datetime
import yaml
import sys
import subprocess


def power_spectral_density(x, time_step, frequency_range = None):
    """
    returns the *single sided* power spectral density of the time trace x which is sampled at intervals time_step
    i.e. integral over the the PSD from f_min~0 to f_max is equal to variance over x


    Args:
        x (array):  timetrace
        time_step (float): sampling interval of x
        freq_range (array or tuple): frequency range in the form [f_min, f_max] to return only the spectrum within this range

    Returns: psd in units of the unit(x)^2/Hz

    """
    N = len(x)
    p = 2 * np.abs(np.fft.rfft(x)) ** 2 / N * time_step

    f = np.fft.rfftfreq(N, time_step)

    if not frequency_range is None:
        assert len(frequency_range) == 2
        assert frequency_range[1] > frequency_range[0]

        bRange = np.all([(f > frequency_range[0]), (f < frequency_range[1])], axis=0)
        f = f[bRange]
        p = p[bRange]

    return f, p




def get_frame_rate(filename):
    """

    Args:
        filename: path to .avi file

    Returns: frame rate in fps

    """
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    rate = out.split('=')[1].strip()[1:-1].split('/')
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1

def get_video_info(filename, verbose = False):
    """
    Args:
        filename: path to video file

    Returns: a dictionary with the video metadata

    """
    v = pims.Video(filename)

    if verbose:
        print(v)

    #JG: v.framerate worked on the Alice computer but not on Eve where I got
    # AttributeError: 'ImageIOReader' object has no attribute 'duration'
    # thus in case v.framerate doesn't work, we try to get the framerate from ffpeg
    try:
        frame_rate = v.frame_rate
    except:
        frame_rate = get_frame_rate(filename)

    # JG: similar for v.duration
    try:
        duration = v.duration
    except:
        duration = len(v)

    video_info = {
        'frame_rate': frame_rate,
        'duration': duration
    }


    return video_info

def get_frames(file_path, frames, gaussian_filter_width=None, roi = None):
    """

    opens the video and returns the requested frames as numpy array

    Args:
        file_path: path to video file
        frames: integer or list of integers of the frames to be returned
        gaussian_filter_width: if not None apply Gaussian filter
        roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!


    """

    images = []
    if not hasattr(frames, '__len__'):
        frames = [frames]

    if not roi is None:
        [roi_center, roi_dimension] = roi

    v = pims.Video(file_path)
    video = rgb2gray_pipeline(v)

    if not gaussian_filter_width is None:
        gaussian_filter_pipeline = pipeline(gaussian_filter)
        video = gaussian_filter_pipeline(video, gaussian_filter_width)


    frame_shape = np.shape(video[frames[0]])



    for frame in frames:

        image = video[frame]

        # reduce image to roi
        if not roi is None:
            [roi_center, roi_dimension] = roi

            image_roi = image[
                        int(roi_center[0] - (roi_dimension[0] - 1) / 2): int(
                            roi_center[0] + (roi_dimension[0] + 1) / 2),
                        int(roi_center[1] - (roi_dimension[1] - 1) / 2): int(roi_center[1] + (roi_dimension[1] + 1) / 2)
                        ]
        else:
            image_roi = image

        images.append(image_roi)

    return images

def load_info(filename):
    """

    Args:
        filename: path to info file

    Returns:
        info as dictionary

    """

    # if position has been extracted previously, there should be an info file
    with open(filename, 'r') as infile:
        info_from_disk = yaml.safe_load(infile)
    return info_from_disk

def reencode_video(filepath, filepath_target = None):
    """

    uses ffmpeg to reencode the video and removes the first second. This is usually
    necessary for the 1200fps videos, which are timestamped weirdly and often have the first second
    corrupted. If you turn this off and receive the error "AVError: [Errno 1094995529] Invalid data found
    when processing input", try turning this on. Will double the runtime of the function.

    Args:
        filepath: path to file of original video
        filepath_target: target file (optional) if None same as input with replacing ".avi" by  "_reencode.avi"

    Returns: nothing but writes reencoded file to disk

    """

    if filepath_target is None:
        filepath_target = filepath.replace('.avi', '_reencode.avi')

    print('start time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    # command string: ffmpeg -i Z:\...\ringdown.avi -s 1 -c copy Z:\...\ringdown_reencode.avi
    # calls ffmpeg, -i specifies input path, -ss 1 cuts first second of video, -c copy copies
    # the input codec and uses it for the output codec, and the last argument is the output file
    # cutting the first second isn't always necessary, but sometimes the videos will not load without it
    cmd_string = "ffmpeg -i " + filepath + " -ss 1 -c copy " + filepath_target
    # performs system (command line) call
    x = os.system(cmd_string)

    print('end time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('wrote:\n{:s}'.format(filepath_target))


def get_position_brightest_px(image, roi = None, verbose = False):
    """


    Args:
        image:  image a 2D array
        roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!

    Returns: the coordinates (row, column) of the brightest pixel in the image

    """

    if verbose:
        print(image)
    # reduce image to roi
    if not roi is None:
        [roi_center, roi_dimension] = roi

        image_roi = image[
                int(roi_center[0] - (roi_dimension[0] - 1) / 2): int(roi_center[0] + (roi_dimension[0] + 1) / 2),
                int(roi_center[1] - (roi_dimension[1] - 1) / 2): int(roi_center[1] + (roi_dimension[1] + 1) / 2)
                ]
    else:
        image_roi = image
        roi_dimension = np.shape(image)

    if verbose:
        print(image_roi)

    pixel_max = np.argmax(image_roi)
    if verbose:
        print('pixel_max', pixel_max, roi_dimension)

    po = np.unravel_index(pixel_max, roi_dimension)
    if verbose:
        print('po', po)
    # po = [po[1], po[0]]  # flip the order to get x, y

    # add the offset from the roi
    if not roi is None:
        offset = [
            int(roi_center[0] - (roi_dimension[0] - 1) / 2),
            int(roi_center[1] - (roi_dimension[1] - 1) / 2)
            ]

        if verbose:
            print('offset', offset)

        po = [
            po[0] + offset[0],
            po[1] + offset[1]
        ]
    else:

        po = list(po)

    return po

def get_position_center_of_mass(image, roi = None, verbose = False):
    """


    Args:
        image:  image a 2D array
        roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!

    Returns: the coordinates (row, column) of the brightest pixel in the image

    """

    if verbose:
        print(image)
    # reduce image to roi
    if not roi is None:
        [roi_center, roi_dimension] = roi

        image_roi = image[
                int(roi_center[0] - (roi_dimension[0] - 1) / 2): int(roi_center[0] + (roi_dimension[0] + 1) / 2),
                int(roi_center[1] - (roi_dimension[1] - 1) / 2): int(roi_center[1] + (roi_dimension[1] + 1) / 2)
                ]
    else:
        image_roi = image
        roi_dimension = np.shape(image)

    if verbose:
        print(image_roi)


    po = center_of_mass(image_roi)
    if verbose:
        print('po', po)
    # po = [po[1], po[0]]  # flip the order to get x, y

    # add the offset from the roi
    if not roi is None:
        offset = [
            int(roi_center[0] - (roi_dimension[0] - 1) / 2),
            int(roi_center[1] - (roi_dimension[1] - 1) / 2)
            ]

        if verbose:
            print('offset', offset)

        po = [
            po[0] + offset[0],
            po[1] + offset[1]
        ]
    else:

        po = list(po)

    return po

def get_position_trackpy(image, po, trackpy_parameters, verbose = False):
    """
    uses trackpy to locate the location of the brightest pixel in the image
    Args:
        image:
        trackpy_parameters:

    Returns:

    """
    locate_info = tp.locate(image, trackpy_parameters['diameter'], minmass=trackpy_parameters['minmass'])
    if verbose:
        print(locate_info)
    pts = locate_info[['x', 'y']].as_matrix()

    if len(pts) == 0:
        pts = None
    elif len(pts) > 0:
        # pick the one that is closest to the original one
        pts = pts[np.argmin(np.array([np.linalg.norm(p - np.array(po)) for p in pts]))]

    return pts

def get_center_of_mass_diff(image, image_ref, xo=None, roi = None, verbose = False):
    """
    calculates the shift of the center of mass of the difference between image and image_ref

    Args:
        image:  image a 2D array
        image_ref: the reference image
        xo: position of first image, since here we just look at relative changes
        roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!

    Returns: the coordinates (row, column) of the brightest pixel in the image

    """

    if verbose:
        print(image)
    # reduce image to roi
    if not roi is None:
        [roi_center, roi_dimension] = roi

        image_roi = image[
                int(roi_center[0] - (roi_dimension[0] - 1) / 2): int(roi_center[0] + (roi_dimension[0] + 1) / 2),
                int(roi_center[1] - (roi_dimension[1] - 1) / 2): int(roi_center[1] + (roi_dimension[1] + 1) / 2)
                ]
    else:
        image_roi = image
        roi_dimension = np.shape(image)

    if verbose:
        print(image_roi)

    if np.sum(image_roi-image_ref)==0:
        po = [0, 0]
    else:
        po = center_of_mass(image_roi-image_ref)
    if verbose:
        print('po', po)
    # po = [po[1], po[0]]  # flip the order to get x, y

    # add the offset from the roi
    if not roi is None:
        offset = [
            int(roi_center[0] - (roi_dimension[0] - 1) / 2),
            int(roi_center[1] - (roi_dimension[1] - 1) / 2)
            ]

        if verbose:
            print('offset', offset)

        po = [
            po[0] + offset[0],
            po[1] + offset[1]
        ]
    else:

        po = list(po)

    return po

def get_position(image, roi, dynamic_roi=False, center_of_mass=False, use_trackpy=False, trackpy_parameters=None):
    """
    gets the position of a bright spot in image useing different methods
    image:  image a 2D array
    roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!
            Note that the order of the dimensions is vertical, horizontal, that is NOT x,y!



    dynamic_roi: if True, dynamically track the roi, ie. the center of the roi of the next frame is the detected position of the currenrt frame.
        If false the roi is static. Is only used if roi is not None

    center_of_mass: calculate the position from the center of mass

    use_trackpy: if True use Trackpy to better localize the brightest point, need to provid trackpy_parameters

    trackpy_parameters: parameters for trackpy (only needed if use_trackpy is True)

    returns: po - the positions found with the different methods
            roi - the current roi
    """

    p_bright = get_position_brightest_px(image, roi)
    po = p_bright

    # update center of roi with current position of bright spot
    if dynamic_roi and not roi is None:
        roi[0] = [int(p_bright[0]), int(p_bright[1])]

    if center_of_mass:
        # get the center of mass
        com = get_position_center_of_mass(image, roi)
        po = po + [com[0], com[1]]

    if use_trackpy:
        p_track = get_position_trackpy(image, p_bright, trackpy_parameters)
        if p_track is None:
            p_track = [None, None]
        po = po + [p_track[0], p_track[1]]  # append point found by trackpy to dataset of current frame

    return po, roi

def extract_motion(filepath, target_path=None,  gaussian_filter_width=2, use_trackpy =False, show_progress = True,
                   trackpy_parameters = None, min_frames = 0, max_frames = None, roi = None, dynamic_roi = False, center_of_mass=False):
    """
    Takes in a bead video, and saves the bead's extracted position in every frame in a csv, and
    (optionally) an uncorrupted version of the video
    filepath: path to an avi file of the bead
    target_path: (optional) if None same as video path

    gaussian_filter_width: width (in pixels) of the gaussian used to smooth the video

    use_trackpy: if True use Trackpy to better localize the brightest point, need to provid trackpy_parameters

    show_progress: if True show progress (use when called from ipython notebook)

    trackpy_parameters: parameters for trackpy (only needed if use_trackpy is True)

    min_frames: first frame to analyze, usefull for debugging, if start from first frame
    max_frames: max number of frames to analyze, usefull for debugging, if none do entire video

    roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!
            Note that the order of the dimensions is vertical, horizontal, that is NOT x,y!

    dynamic_roi: if True, dynamically track the roi, ie. the center of the roi of the next frame is the detected position of the currenrt frame. If false the roi is static

    center_of_mass: calculate the position from the center of mass

    returns: path to .csv file
    """


    if target_path is None:
        target_path = os.path.dirname(filepath)

    assert os.path.isdir(target_path)

    if use_trackpy:
        assert 'diameter' in trackpy_parameters
        if not 'minmass' in trackpy_parameters:
            trackpy_parameters['minmass'] = None
        trackpy_parameters['missing_frames'] = []

    target_filename = os.path.join(target_path, os.path.basename(filepath).replace('.avi',  '_data_globalmax.csv'))

    v = pims.Video(filepath)
    processed_v = rgb2gray_pipeline(v)

    gaussian_filter_pipeline = pipeline(gaussian_filter)
    filtered = gaussian_filter_pipeline(processed_v, gaussian_filter_width)

    if max_frames is None:
        max_frames=len(filtered)
    else:
        # make sure max_frames is integer
        max_frames =int(max_frames)

    if show_progress:
        f = FloatProgress(min=min_frames, max=max_frames)  # instantiate the bar
        display(f)  # display the bar
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('start time:\t{:s}'.format(start_time))



    image_size = np.shape(filtered[0]) # size of image
    # find the maximum brightness pixel for every frame, corresponding to some consistent point at the bead
    max_coors = []
    skipped_frames_idx = []


    if not roi is None:

        [roi_center, roi_dimension] = roi

        for i in [0, 1]:
            # assert that roi fits in the image
            assert roi_center[i] + (roi_dimension[i]+1) / 2 <= image_size[i]
            assert roi_center[i] - (roi_dimension[i]+1) / 2 >= 0

            # assert that roi_dimension are odd
            assert roi_dimension[i] % 2 == 1

    cols = ['y bright', 'x bright']
    if center_of_mass:
        cols = cols + ['x com', 'y com']
    if use_trackpy:
        cols = cols + ['x tp', 'y tp']
        
    # we use range in this loop so that we can catch the CannotReadFrameError in the line where we try to access the next image
    for index in range(min_frames, max_frames):
        try:
            image = filtered[index]
        except CannotReadFrameError:
            skipped_frames_idx.append(index)

        po, roi = get_position(image, roi, dynamic_roi=dynamic_roi, center_of_mass=center_of_mass, use_trackpy=use_trackpy,
                         trackpy_parameters=trackpy_parameters)

        if po is None:
            trackpy_parameters['missing_frames'].append(index)
        else:
            max_coors.append(po)


        if index % 1000 == 0:
            if show_progress:
                f.value += 1000

        if not max_frames is None:
            if index>max_frames:
                break

    df = pd.DataFrame(max_coors, columns=cols)
    df.to_csv(target_filename)
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('end time:\t{:s}'.format(end_time))
    print('file written to:\n{:s}'.format(target_filename))


    info = {
        'filename_xy_position': target_filename,
        'image_size':np.shape(filtered[0]),
        'gaussian_filter_width':gaussian_filter_width,
        'N_frames':len(filtered),
        'use_trackpy':use_trackpy,
        'center_of_mass':center_of_mass,
        'start_time':start_time,
        'end_time':end_time,
        'max_frames':max_frames,
        'skipped_frames_idx':skipped_frames_idx,
        'N_skipped_frames': len(skipped_frames_idx),
        'roi' : roi
    }

    # convert time back to datetime so that we can calculate the duration
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    info['extract. duration (min)'] = (end_time - start_time).seconds / 60


    if use_trackpy:
        info.update({
            'trackpy_parameters': trackpy_parameters
        })

    if info['N_skipped_frames']>100:
        print('WARNING: more than 100 frames corrupted in this video!!')

    return info


def tracking_error(x_ref, x):
    """

        returns: direct error and error between differentials (gets rid of a global off set)
    """
    err = np.std(x_ref - x)
    err_diff = np.std(np.diff(x_ref) - np.diff(x))
    return err, err_diff


def calc_tracking_error(images, center=None, roi=None, dynamic_roi=False, center_of_mass=False, use_trackpy=False,
                        trackpy_parameters=None):
    data = []
    for image, c in zip(images, center):
        po, roi = get_position(image, roi, dynamic_roi=dynamic_roi, center_of_mass=center_of_mass,
                                  use_trackpy=use_trackpy, trackpy_parameters=trackpy_parameters)
        data.append(c + po)

    cols = ['xo', 'yo', 'y bright', 'x bright']
    if center_of_mass:
        cols = cols + ['y com', 'x com']
    if use_trackpy:
        cols = cols + ['x track', 'y track']
    data = pd.DataFrame(data, columns=cols)

    return data

def load_xy_time_trace(filepath, center_at_zero = True):
    """
    Takes in a filepath to a csv containing the bead positions

    Args:
        filepath: filepath to a csv containing the bead positions
        center_at_zero: is true substract mean, else return raw data

    Returns: the x and y position as a Nx2 matrix where N is the number of datapoints and the first column is x and the second y

    """
    #load data
    xy = pd.read_csv(filepath)
    xy = xy.as_matrix()

    xy = xy[:,1:] # drop first column since this is just the index

    if center_at_zero:
        xy = xy - np.mean(xy, axis=0)

    return xy

if __name__ == '__main__':

    # #A) ======== testing get_brightest_px ==========
    #
    # # simple test no roi
    # # po = np.random.randint(3,8, size=2)
    # #
    # # print('initial', po)
    # # image = np.zeros([11, 11])
    # # image[po[0], po[1]] = 255
    # #
    # # po = get_brightest_px(image, roi=None)
    # # print('found', po)
    #
    #
    # # test with roi
    # po = np.random.randint(3,8, size=2)
    #
    # print('initial', po)
    # image = np.zeros([11, 11])
    # image[po[0], po[1]] = 255
    #
    # po = get_brightest_px(image, roi=[po, [5, 3]], verbose=True)
    # print('found', po)
    #
    #
    # #  ======== end testing  ========


    #B) ======== testing center_of_mass ==========



    # # test with roi
    # v = False
    # po = np.random.randint(3,8, size=2)
    #
    # print('initial', po)
    # image = np.ones([11, 11])*3
    # image[po[0], po[1]] = 255
    #
    # roi=[po, [5, 3]]
    # roi = [[po[0]+1, po[1]], [5, 3]]
    #
    # image = image- np.mean(image)
    #
    # po = get_position_brightest_px(image, roi=roi, verbose=v)
    # print('found', po)
    # com = get_position_center_of_mass(image, roi=roi, verbose=v)
    # print('found com', com)
    #
    # #  ======== end testing  ========

    print(tp.__version__)