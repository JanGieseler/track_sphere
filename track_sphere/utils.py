import os, yaml
import numpy as np
import cv2 as cv
# import colors
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from glob import glob
import operator
from functools import reduce
from scipy import optimize

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

def get_wrap_angle(angles, bins=2000, navrg=1, n_smooth_dist = 1, plot_distibutions_axes=None):
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

    xmin = np.where(x[0] == x[0].min())[0]
    if len(xmin) == 1:
        wrap_angle = x1[np.argmin(x0)]
    else:
        # there is more than one value with the smalles value,
        # then we look for the larges range of continuous values and take the average
        range_pairs = [i for i, df in enumerate(np.diff(xmin)) if df > 1]
        imax = np.argmax(np.diff(xmin[range_pairs]))
        wrap_angle = np.mean(x[1][[xmin[range_pairs][imax], xmin[range_pairs][imax + 1]]])


    if plot_distibutions_axes is not None:
        plot_distibutions_axes.semilogy(x1, x0/np.sum(x0))

    return wrap_angle


def get_rotation_frequency_fit_slope(data, info, n_avrg=1, n_avrg_unwrapped=1, wrap_angle=None, return_figure=False, axes=None, nmax=500):
    """
    calculate the rotation frequency from a time trace of angle data, the assumption is that the rotation is constant
    Args:
        data:
        info:
        n_avrg: nmber of points used for smoothing the data
        wrap_angle: if None find wrap_angle automatically otherwise wrap angles at this value

    Returns:

    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(8 * 2,6))
    else:
        fig = None

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
    t = np.arange(len(rot_angle))* time_step * n_avrg_unwrapped
    fit, cov = np.polyfit(t, rot_angle/360, 1, cov=True)

    freq = fit[0]

    # err = np.std(rot_angle - np.poly1d(fit)(t)) / np.mean(rot_angle) * fit[0]  # relative error of fit
    err = np.sqrt(np.diag(cov))[0]



    return_dict = {'freq':freq, 'err': err,
                   'timestamp':timestamp,
                   'n_avrg':n_avrg, 'n_avrg_unwrapped':n_avrg_unwrapped
                   }

    print(return_dict)

    if return_figure:
        axes[0].plot(t[0:nmax], rot_angle[0:nmax]/360, '.')
        axes[0].plot(t[0:nmax],  np.poly1d(fit)(t[0:nmax]), '-', linewidth = 3)
        axes[1].plot(t, rot_angle/360, '.')
        axes[1].plot(t, np.poly1d(fit)(t), '-', linewidth = 3)

        axes[0].set_ylabel('rotations')
        axes[0].set_xlabel('time (s)')
        axes[1].set_ylabel('rotations')
        axes[1].set_xlabel('time (s)')

        return fig, axes, return_dict
    else:
        return return_dict



def get_calibration_factor(data, particle_diameter, verbose=False):
    """
    calculates the calibration factor assuming that the particle is roughly spherical
    Args:
        data: pandas dataframe with 'ellipse a' and 'ellipse b' columns
        particle_diameter: diameter of partile in um
        verbose:

    Returns:

    """

    x = np.sqrt(data['ellipse a']*data['ellipse b'])
    if verbose:
        print('calibration factor: (um/px)', particle_diameter/np.mean(x))
    return {'calibration factor: (um/px)': particle_diameter/np.mean(x)}

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

    counts, bins = np.histogram(x, bins=100, density=False)

    if exclude_percent is not None:
        # get the min and max angle from the histogram
        angle_min = min(bins[:-1][counts / np.max(counts) > 0.1])
        angle_max = max(bins[:-1][counts / np.max(counts) > 0.1])

    # if jumps are bigger than 20% then say that this is a discontinuity
    angle_jump = (angle_max - angle_min) / 5
    time_step = 1. / info['info']['FrameRate']

    # select all the angles between angle_min and angle_max
    boolean_selector = np.logical_and(x >= angle_min, x <= angle_max)

    # figure out the orientation
    left = np.logical_and(boolean_selector, np.hstack([np.diff(x), 0]) < angle_jump)
    right = np.logical_and(boolean_selector, np.hstack([np.diff(x), 0]) > angle_jump)

    boolean_selector = left if sum(left) > sum(right) else right

    direction = 'left' if sum(left) > sum(right) else 'right'


    selector = np.where(boolean_selector)[0]
    # find all the values where data is not continuous and arrange in pairs
    range_pairs = [i for i, df in enumerate(np.diff(selector)) if df != 1]
    range_pairs = np.hstack([-1, range_pairs, len(selector) - 1])  # add first and last elements
    range_pairs = np.vstack([range_pairs[:-1] + 1, range_pairs[1:]]).T
    # remove the elements where there is only a single value
    range_pairs = [i for i in range_pairs if np.diff(i) > 1]

    # now calculate the freq from the slope of the continuous ranges of data
    def fdiff(i):
        # define helper function
        yo = x[range(selector[i][0], selector[i][1])]
        yo = np.unwrap(yo, angle_jump)
        return np.diff(yo) / time_step / 360

    freqs = [fdiff(i) for i in range_pairs]
    freqs = np.hstack(freqs) # turn into 1D array

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

    return_dict = {'mean':np.mean(freqs), 'std': np.std(freqs),
                   'exclude_percent':exclude_percent,
                   'angle_min':angle_min, 'angle_max':angle_max
                   }

    if return_figure:
        t = time_step * np.arange(nmax)
        counts, bins, _ = axes[1].hist(x, bins=100, log=True, alpha=0.3)
        axes[1].hist(x[selector], bins=bins, log=True, alpha=0.3)
        _, bins, _ = axes[2].hist(np.diff(x) / time_step / 360, bins=100, log=True, alpha=0.3)
        axes[2].hist(freqs, bins=bins, log=True, alpha=0.3)

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
            if direction == 'left':
                # axes[0].plot(to, max(yo)+fmean(i)*360*(to-to[0]), 'k-')
                axes[0].plot(to, max(yo) + np.mean(fdiff(i)) * 360 * (to - to[0]), 'k-')
            else:
                axes[0].plot(to, min(yo) + np.mean(fdiff(i)) * 360 * (to - to[0]), 'k-')


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

        return fig, axes, return_dict
    else:
        return return_dict

def get_position_file_names(source_folder_positions, method, tag = 'Sample_6_Bead_1', runs = None):
    """

    Args:
        source_folder_positions: name of folder
        method: extraction method for position information
        runs: if not non give a list of integers with the run numbers

    Returns: all the filenames in the folder source_folder_positions sorted by run id

    """
    # get all the files and sort them by the run number
    position_file_names = sorted([os.path.basename(f) for f in glob(source_folder_positions + '*-'+method+'.dat')])

    if runs is not None:
        position_file_names = [f for f in position_file_names if int(f.split('-')[0].split(tag)[1].split('_')[1]) in runs]
    position_file_names = sorted(position_file_names,
                                 key=lambda f: int(f.split('-')[0].split(tag)[1].split('_')[1]))
    return position_file_names

def get_mode_frequency_fft(data, mode, info, return_figure=False, interval_width=None, interval_width_zoom=0.1, fo=None,
                           verbose=False, n_smooth=None, method='fit_ellipse'):

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
        n_smooth: if not None avrg n_smooth values to smoothen the spectra
        aliasing: if True, freq is aliased and we calculate the real freq. as fps - f_max
    Returns:

    """
    time_step = 1 / info['info']['FrameRate']
    freqs = {}

    # if the target freq is larger than the Nyquist freq., the actual freq gets folded back
    if fo >info['info']['FrameRate']/2:
        aliasing = True
    else:
        aliasing = False


    if method=='fit_ellipse':
        if mode == 'r':
            x = data['ellipse angle']
        elif mode == 'm':
            x = data['ellipse y']
        elif mode == 'r-unwrap':
            x = data['ellipse angle']
            x = np.unwrap(x, get_wrap_angle(x))
        elif mode == 'z':
            x = data['ellipse x'] * data['ellipse y'] * np.pi
        else:
            x = data['ellipse ' + mode]

    elif method.lower() == 'bright px':
        if len(mode) == 1:
            x = data['bright px ' + mode]
        else:
            # the last character should indicate the direction we use,
            # e.g. z-x or zx to calculate the z frequency from the x direction timetrace
            x = data['bright px ' + mode[-1]]

    x = x-np.mean(x)  # make zero mean

    f, p = power_spectral_density(x, time_step, frequency_range=None)


    if aliasing:
        fo = info['info']['FrameRate']-fo

    if interval_width is None:
        frequency_range = (min(f[1:]), max(f))  # for the minimum ignore the first value because this is DC
    else:
        frequency_range = (fo - interval_width / 2, fo + interval_width / 2)

    if n_smooth is None:
        f2, p2 = f, p
    else:
        f2 = avrg(f, n_smooth)
        p2 = avrg(p, n_smooth)

    # pick the range of interest
    bRange = np.all([(f2 > frequency_range[0]), (f2 < frequency_range[1])], axis=0)
    F = f2[bRange]
    P = p2[bRange]

    freqs[mode] = F[np.argmax(P)]

    df = np.mean(np.diff(F))

    frequency_range_zoom = (freqs[mode] - 0.5 * interval_width_zoom, freqs[mode] + 0.5 * interval_width_zoom)
    bRange_zoom = np.all([(F >= frequency_range_zoom[0]), (F <= frequency_range_zoom[1])], axis=0)

    F_zoom = F[bRange_zoom]
    P_zoom = P[bRange_zoom]

    if aliasing:
        freqs[mode] = info['info']['FrameRate']-freqs[mode]
        F_zoom = info['info']['FrameRate']-F_zoom


    freqs[mode + '_power'] = np.sum(P_zoom) * df


    if verbose:
        print(mode + ': ', freqs[mode])

    if return_figure:
        ## plot
        fig, axes = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(8 * 3, 4))
        axes[0].semilogy(F, P / max(P))

        axes[1].plot(F_zoom, P_zoom, 'o')

        # zoom plot
        if n_smooth is not None:
            # pick the range of interest
            bRange = np.all([(f > frequency_range[0]), (f < frequency_range[1])], axis=0)
            F = f[bRange]
            P = p[bRange]

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


def power_to_energy_K(x, radius, frequency, calibration_factor, density=7600):
    """

    scaled the input power to physical units

    Args:
        x: energy in px^2
        radius: in um
        frequency: in Hertz
        calibration_factor: in um/px
        density: in kg/m^3

    Returns:

    """
    mass = density*4*np.pi/3 * radius**3*1e-18
    kB = 1.38e-23
    return x*calibration_factor**2*1e-12*mass*(2*np.pi*frequency)**2/kB

def get_ampfreqphase_FFT(qx, dt, n0 = 0, f_range = None, return_Spectra = False):
    '''
    returns estimate of amplitdue, frequency and phase from FFT

    [ax, wx, phi] = get_ampfreqphase_FFT(qx, dt,n0 = 0, f_range=None, return_Spectra = False)
    [ax, wx, phi], [Fx, Ax] = get_ampfreqphase_FFT(qx, dt,n0 = 0, f_range=None, return_Spectra = True)
    input:
        qx: time trace  sampled at intervals dt
        dt: sampling interval

    input (optional):
        n0 = t0/dt: index of time zero
        f_range = [f_x, df]: frequency is looked in intervals f_x +-df respectively
        return_Spectra = True/False: returns spectra over range f_range in addition to [phi, ax, fx]

    output:
        dominant angular frequency, amplitude at that frequency and phase
        method: get fourier component of max signals
    '''

    n = len(qx)
    f = np.fft.fftfreq(n, dt)[0:int(n/2)]

    # look for max frequencies only in certain range
    if f_range is None:
        irange_x = np.arange(int(n/2))
    else:
        [f_x, df] = f_range
        imin = np.argwhere(f >= f_x-df)[0,0]
        imax = np.argwhere(f <= f_x+df)[-1,0] + 1
        irange_x = np.arange(imax-imin+1)+imin

    # convert to int (from float)
    irange_x = [int(x) for x in irange_x]

    # Fourier transforms (remove offset, in case there is a large DC)
    Ax = np.fft.fft(qx-np.mean(qx))[irange_x] / n*2
    Fx = f[irange_x]

    # frequency and amplitude x
    i_max_x = np.argmax(np.abs(Ax))
    fx = Fx[i_max_x]
    ax = np.abs(Ax[i_max_x])
    # phase
    phi = np.angle(Ax[i_max_x] * np.exp(-1j *2 * np.pi * fx * n0))

    if return_Spectra == True:
        return [ax, 2*np.pi*fx, phi], [Fx, Ax]
    else:
        return [ax, 2*np.pi*fx, phi]


def fit_exp_decay(t, y, offset=False, verbose=False):
    """
    fits the data to a decaying exponential, with or without an offset
    Args:
        t: x data
        y: y data
        offset: False if fit should decay to y=0, True otherwise
        verbose: prints results to screen

    Returns: fit parameters, either [ao, tau, offset] if offset is True, or or [ao, tau] if offset is False
            ao: amplitude above offset (or zero if offset is False)
            tau: decay parameter
            offset: asymptotic value as t->INF

    """
    if verbose:
        print(' ======= fitting exponential decay =======')

    init_params = estimate_exp_decay_parameters(t, y)
    if offset:
        fit = optimize.curve_fit(exp_offset, t, y, p0=init_params)
    else:
        fit = optimize.curve_fit(exp, t, y, p0=init_params[0:-1])
    if verbose:
        print(('optimization result:', fit))

    return fit


def estimate_exp_decay_parameters(t, y):
    '''
    Returns an initial estimate for exponential decay parameters. Meant to be used with optimize.curve_fit.
    Args:
        t: x data
        y: y data

    Returns: fit parameter estimate, either [ao, tau, offset] if offset is True, or or [ao, tau] if offset is False
            ao: amplitude above offset (or zero if offset is False)
            tau: decay parameter
            offset: asymptotic value as t->INF

    '''
    offset = y[-1]

    total_amp = y[0]
    ao = total_amp - offset
    decay = t[np.argmin(np.abs(
        y - (total_amp + offset) / 2))]  # finds time at which the value is closest to midway between the max and min

    return [ao, decay, offset]


def exp(t, *params):
    '''
    Exponential decay: ao*E^(t/tau)
    '''
    ao, tau = params

    return np.exp(-t / tau) * ao


def exp_offset(t, *params):
    '''
    Exponential decay with offset: ao*E^(t/tau) + offset
    '''
    ao, tau, offset = params
    return np.exp(-t / tau) * ao + offset

# def sequence_pairs(x):
#     """
#     find all the values where data is continuous and arrange in pairs
#     Args:
#         x: a list of indecies
#
#     Returns:
#
#     """
#     range_pairs = [i for i, df in enumerate(np.diff(x)) if df > 1]
#     range_pairs = np.hstack([-1, range_pairs, len(x) - 1])  # add first and last elements
#     range_pairs = np.vstack([range_pairs[:-1] + 1, range_pairs[1:]]).T
#
#     return range_pairs
#

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

