import os, yaml
import numpy as np
import cv2 as cv
# import colors
import pandas as pd
import datetime
from glob import glob



def roi_2_roi_tlc(roi):
    """

    converts the roi with x,y,w,h that is centered at x,y to a roi that is defined by the top-left-corner

    Args:
        roi: (x, y, w, h)

    Returns:
        roi_tlc: (r, c, w, h)

    """
    return (roi[1] - int(roi[3] / 2), roi[0] - int(roi[2] / 2), roi[3], roi[2])

def lrc_from_features(features, winSize, num_features=None):
    """
    gets the lower right corner from the center point of a feature

    Args:
        features: list with features of length (num_features x 4)
                    the four numbers are angle, size, x, y

    Returns: positions as a array with shape = (num_features, 2)

    """
    if num_features is None:
        num_features = int(len(features)/4)

    positions = np.reshape(features, [num_features,  4])[:,:2].astype(int)
    positions[:, 0] -= int(winSize[0] / 2)
    positions[:, 1] -= int(winSize[1] / 2)

    return positions

def points_from_blobs(blobs, num_blobs=None):
    """
    gets the center points from fit_blobs output data

    Args:
        blobs: list with blobs data of length (num_blobs x 6)
                    the four numbers are threshold, ellipse center (x, y), size (a,b) and angle

    Returns: positions as a array with shape = (num_blobs, 2)

    """
    if num_blobs is None:
        num_blobs = int(len(blobs)/6)

    positions = np.reshape(blobs, [num_blobs,  6])[:,1:3].astype(int)
    # positions[:, 0] -= int(winSize[0] / 2)
    # positions[:, 1] -= int(winSize[1] / 2)

    return positions

def select_initial_points(file_in, frame = 0):
    """
    loads frame and returns points that have been selected with a double click
    Args:
        file_in: path to video file
        frame: index of frame

    Returns: positions of selected points

    """


    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        if event == cv.EVENT_LBUTTONDBLCLK:

            cv.circle(frame,(x,y),5,(255,0,0),-1)
            positions.append([x,y])
            print('position', x, y)

    title = 'select blob with double-click, to finish press ESC'
    # bind the function to window
    cv.namedWindow(title)
    cv.setMouseCallback(title,draw_circle)

    # load frame of video
    cap = cv.VideoCapture(file_in)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame)  # set the starting frame for reading to frame
    ret, frame = cap.read()

    positions = []

    while(1):
        cv.imshow(title,frame)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv.destroyAllWindows()

    return positions

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

def avrg(x, n=10):
    """
    averages piecewise blocks of n data points into a new data point
    Args:
        x:
        n:

    Returns:

    """
    m = int(len(x)/n)
    return np.mean(x[0:n*m].reshape([m, n]), axis=1)

def get_wrap_angle(angles, bins=2000, navrg=50):
    """
    calcuates the optimal wrap angle based on the histogram of the differential of the angles
    the histogram has a peak around zero and 180 deg
    Args:
        angles:
        bins:
        navrg:

    Returns: optimal wrap angle

    """
    x = np.histogram(abs(np.diff(angles)), bins=bins)
    x0 = avrg(x[0], n=navrg)
    x1 = avrg(x[1], n=navrg)
    wrap_angle = x1[np.argmin(x0)]
    return wrap_angle


def get_rotation_frequency(data, info, n_avrg=20):
    """
    calculate the rotation frequency from a time trace of angle data, the assumption is that the rotation is constant
    Args:
        data:
        info:

    Returns:

    """
    time_step = 1. / info['info']['FrameRate']
    rot_angle = np.unwrap(data['ellipse angle'], discont=get_wrap_angle(data['ellipse angle']))
    freqs = np.diff(rot_angle) / (360 * time_step)

    rot_angle = avrg(rot_angle, n=n_avrg)
    freqs = np.diff(rot_angle) / (360 * time_step * n_avrg)

    return np.mean(freqs), np.std(freqs), info['info']['File_Modified_Date_Local'], n_avrg

def get_position_file_names(source_folder_positions, method):
    """

    Args:
        source_folder_positions: name of folder
        method: extraction method for position information

    Returns: all the filenames in the folder source_folder_positions

    """
    # get all the files and sort them by the run number
    position_file_names = sorted([os.path.basename(f) for f in glob(source_folder_positions + '*-'+method+'.dat')])
    position_file_names = sorted(position_file_names, key=lambda f: int(f.split('-')[0].split('Bead_')[1].split('_')[0]))
    return position_file_names

if __name__ == '__main__':
    folder_in = '../example_data/'
    filename_in = '20171207_magnet.avi'

    folder_in = '../raw_data/'
    # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    filename_in = '20180529_Sample6_bead_1_direct_thermal_01c.avi'
    # filename_in = '20180523_Sample6_bead_1_direct_thermal_03_reencode.avi'
    # filename_in = '20180524_Sample6_bead_1_direct_thermal_pump_isolated_10b.avi'


    file_in = os.path.join(folder_in, filename_in)


    ffmpeg_segment_video(file_in, 100)


    # fix key
    # file_out = file_in.replace('.avi', '-fixed.avi')
    # cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy '
    # cmd_string += ' -reset_timestamps 1 ' + file_out
    #
    # print(cmd_string)
    # x = os.system(cmd_string)

