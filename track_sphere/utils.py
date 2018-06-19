import os, yaml
import numpy as np
import cv2 as cv
# import colors
import pandas as pd
import datetime
import matplotlib.pyplot as plt
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

def get_wrap_angle(angles, bins=2000, navrg=50, n_smooth_dist = 50, plot_distibutions_axes=None):
    """
    calcuates the optimal wrap angle based on the histogram of the differential of the angles
    the histogram has a peak around zero and 180 deg
    Args:
        angles:
        bins: number of bins in histogram
        navrg: number of points used to smoothen the data
        n_smooth_dist: number of avrg to smoothen distribution data

    Returns: optimal wrap angle

    """


    x = np.histogram(abs(np.diff(avrg(angles, n=navrg))), bins=bins)


    x0 = avrg(x[0], n_smooth_dist)
    x1 = avrg(x[1], n_smooth_dist)
    wrap_angle = x1[np.argmin(x0)]


    if plot_distibutions_axes is not None:
        plot_distibutions_axes.semilogy(x1, x0/np.sum(x0))

    return wrap_angle


def get_rotation_frequency_old(data, info, n_avrg=20 ,n_avrg_unwrapped=20, wrap_angle=None):
    """
    calculate the rotation frequency from a time trace of angle data, the assumption is that the rotation is constant
    Args:
        data:
        info:
        n_avrg: nmber of points used for smoothing the data
        wrap_angle: if None find wrap_angle automatically otherwise wrap angles at this value

    Returns:

    """

    if isinstance(info, dict):
        time_step = 1. / info['info']['FrameRate']
        timestamp = info['info']['File_Modified_Date_Local']
    else:
        time_step = info
        timestamp = None

    if wrap_angle is None:
        wrap_angle = get_wrap_angle(data['ellipse angle'], navrg=n_avrg)
    rot_angle = avrg(data['ellipse angle'], n=n_avrg)
    rot_angle = np.unwrap(rot_angle, discont=wrap_angle)
    rot_angle = avrg(rot_angle, n=n_avrg_unwrapped)
    freqs = np.diff(rot_angle) / (360 * time_step * n_avrg_unwrapped)

    return np.mean(freqs), np.std(freqs), timestamp, n_avrg, n_avrg_unwrapped


def get_rotation_frequency(data, info, return_figure=False, exclude_percent=None, angle_min=50, angle_max=130, nmax=100, axes=None):
    """
    calculate the rotation frequency from a time trace of angle data, the assumption is that the rotation is constant
    Args:
        data:
        info:
        return_figure:
        exclude_percent: value between 0 and 1, all the angles that are below this percentage wrt the max angle count are excluded
        angle_min: minimum angle to be taken into account, if exculude_percent is not None, this value is ignored
        angle_max: maximum angle to be taken into account, if exculude_percent is not None, this value is ignored
        nmax:
        axes:

    Returns:

    """

    if axes is None:
        fig, axes = plt.subplots(1, 3, sharey=False, sharex=False, figsize=(8 * 3, 8))
    else:
        fig = None

    x = data['ellipse angle'].as_matrix()

    counts, bins, _ = axes[1].hist(x, bins=100, log=True, density=False, alpha=0.3)


    if exclude_percent is not None:
        # get the min and max angle from the histogram
        angle_min = min(bins[:-1][counts / np.max(counts) > 0.1])
        angle_max = max(bins[:-1][counts / np.max(counts) > 0.1])

    angle_jump = (angle_max - angle_min) / 2
    time_step = 1. / info['info']['FrameRate']

    def boolean_selector(x, direction):
        if direction=='left':
            return

    # select all the angles between angle_min and angle_max
    boolean_selector = np.logical_and(x >= angle_min, x <= angle_max)

    # figure out the orientation
    left = np.logical_and(boolean_selector, np.hstack([np.diff(x), 0]) < 0)
    right = np.logical_and(boolean_selector, np.hstack([np.diff(x), 0]) > 0)
    boolean_selector = left if sum(left)>sum(right) else right


    boolean_selector = np.logical_and(boolean_selector, np.hstack([np.diff(x), 0]) < 0)

    selector = np.where(boolean_selector)[0]
    # find all the values where data is not continuous and arrange in pairs
    range_pairs = [i for i, df in enumerate(np.diff(selector)) if df != 1]
    range_pairs = np.hstack([-1, range_pairs, len(selector) - 1])  # add first and last elements
    range_pairs = np.vstack([range_pairs[:-1] + 1, range_pairs[1:]]).T
    # remove the elements where there is only a single value
    range_pairs = [i for i in range_pairs if np.diff(i) > 1]

    # now calculate the freq from the slope of the continuous ranges of data
    def fmean(i):
        # define helper function
        yo = x[range(selector[i][0], selector[i][1])]
        yo = np.unwrap(yo, angle_jump)
        return np.mean(np.diff(yo)) / time_step / 360

    freqs = [fmean(i) for i in range_pairs]
    # now calculate the freq from the slope of the continuous ranges of data - by fitting
    # def linfit(i):
    #     # define helper function
    #     # xo = np.arange(selector[i[0]], selector[i[1]])
    #     # yo = x[xo]
    #     to = t[range(selector[i][0], selector[i][1])]
    #     yo = x[range(selector[i][0], selector[i][1])]
    #     fit = np.polyfit(to, yo, 1)
    #     return fit

    # freqs = [linfit(i)[1] for i in range_pairs]

    if return_figure:
        t = time_step * np.arange(nmax)
        axes[1].hist(x[selector], bins=bins, log=True, density=False, alpha=0.3)
        _, bins, _ = axes[2].hist(np.diff(x) / time_step / 360, bins=100, log=True, density=False, alpha=0.3)
        axes[2].hist(freqs, bins=bins, log=True, density=False, alpha=0.3)

        axes[0].plot(t, x[0:nmax], 'o')
        for i in range_pairs:
            if selector[i[1]] > nmax:
                break
            to = t[range(selector[i][0], selector[i][1])]
            yo = x[range(selector[i][0], selector[i][1])]
            axes[0].plot(to, yo, 'x')

            # fit = np.polyfit(to, yo, 1)
            # fit = linfit(i)
            # axes[0].plot(to, np.poly1d(fit)(to), 'k-')
            axes[0].plot(to, max(yo)+fmean(i)*360*(to-to[0]), 'k-')


        axes[0].set_title('angle (deg)')
        axes[0].set_xlabel('time (selector)')
        axes[1].set_title('angle (deg)')
        axes[1].set_xlabel('angle (deg)')
        axes[1].set_title('probability (counts)')
        axes[2].set_xlabel('freq (Hz)')
        axes[2].set_title('probability (counts)')

        # mark the phase boundaries
        axes[0].plot([0, time_step * nmax], [angle_min, angle_min], 'k--')
        axes[0].plot([0, time_step * nmax], [angle_max, angle_max], 'k--')

        return fig, axes, np.mean(freqs), np.std(freqs)
    else:
        return np.mean(freqs), np.std(freqs)

def get_position_file_names(source_folder_positions, method):
    """

    Args:
        source_folder_positions: name of folder
        method: extraction method for position information

    Returns: all the filenames in the folder source_folder_positions sorted by run id

    """
    # get all the files and sort them by the run number
    position_file_names = sorted([os.path.basename(f) for f in glob(source_folder_positions + '*-'+method+'.dat')])
    position_file_names = sorted(position_file_names, key=lambda f: int(f.split('-')[0].split('Bead_')[1].split('_')[0]))
    return position_file_names


def get_mode_frequency(data, mode, info, return_figure=False, interval_width=None, interval_width_zoom=0.1, fo=None,
                       verbose=False):

    """

    Args:
        data: data as a pandas frame obtained from ellipse fitting
        mode: one of the modes, x y y r
        info: info file
        return_figure: if true return figure and axes objects
        interval_width: witdth of range where to look for peak, if None take center of full range
        interval_width_zoom:(for plotting)
        fo: center of range where to look for peak, if None take center of full range
        verbose:

    Returns:

    """
    time_step = 1 / info['info']['FrameRate']
    freqs = {}

    if mode == 'r':
        x = data['ellipse angle']
    elif mode == 'z':
        x = data['ellipse x'] * data['ellipse y'] * np.pi
    else:
        x = data['ellipse ' + mode]

    f, p = power_spectral_density(x, time_step, frequency_range=None)

    if interval_width is None:
        frequency_range = (min(f), max(f))
    else:
        frequency_range = (fo - interval_width / 2, fo + interval_width / 2)

    # pick the range of interest
    bRange = np.all([(f > frequency_range[0]), (f < frequency_range[1])], axis=0)
    F = f[bRange]
    P = p[bRange]

    freqs[mode] = F[np.argmax(P)]

    if verbose:
        print(mode + ': ', freqs[mode])

    if return_figure:
        ## plot
        fig, axes = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(8 * 3, 4))
        axes[0].plot(F, P / max(P))

        # zoom plot
        frequency_range_zoom = (freqs[mode] - 0.5 * interval_width_zoom, freqs[mode] + 0.5 * interval_width_zoom)
        bRange_zoom = np.all([(F >= frequency_range_zoom[0]), (F <= frequency_range_zoom[1])], axis=0)
        F_zoom = F[bRange_zoom]
        P_zoom = P[bRange_zoom]

        axes[1].plot(F_zoom, P_zoom)
        axes[1].set_title(mode + ' axis')

        for a in axes:
            a.set_xlabel('frequency (Hz)')
        return fig, axes, freqs
    else:
        return freqs

if __name__ == '__main__':
    folder_in = '../example_data/'
    filename_in = '20171207_magnet.avi'

    folder_in = '../raw_data/'
    # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    filename_in = '20180529_Sample6_bead_1_direct_thermal_01c.avi'
    # filename_in = '20180523_Sample6_bead_1_direct_thermal_03_reencode.avi'
    # filename_in = '20180524_Sample6_bead_1_direct_thermal_pump_isolated_10b.avi'


    file_in = os.path.join(folder_in, filename_in)




    # fix key
    # file_out = file_in.replace('.avi', '-fixed.avi')
    # cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy '
    # cmd_string += ' -reset_timestamps 1 ' + file_out
    #
    # print(cmd_string)
    # x = os.system(cmd_string)

