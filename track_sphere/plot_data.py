import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Rectangle, Circle

import numpy as np
from track_sphere.read_write import grab_frame

from track_sphere.utils import power_spectral_density, avrg, get_wrap_angle
from track_sphere.plot_utils import annotate_frequencies

def plot_ellipse_spectra(data, info, annotation_dict={}, freq_range=None, n_avrg=None, plot_type = 'lin', verbose=False, normalize=True, return_data=False, axes = None):
    """

    Args:
        data:
        info:
        annotation_dict:
        freq_range:
        n_avrg:
        plot_type:

    Returns:

    """

    coordinates = [['x', 'y', 'area'], ['a', 'b', 'angle']]
    feature = 'ellipse'

    time_step = 1. / info['info']['FrameRate']
    axes_shape = np.shape(coordinates)

    if axes is None:
        fig, axes = plt.subplots(axes_shape[0], axes_shape[1], sharey=False, sharex=False,
                                 figsize=(8 * axes_shape[1], 3 * axes_shape[0]))
    else:
        fig = None




    if return_data:
        #containers for the data
        p_list = []
    modes = {}
    for i, row in enumerate(zip(coordinates, axes)):

        #     print('row', row)
        c_row, a_row = row
        for c, ax in zip(c_row, a_row):

            if verbose:
                print('plotting ' + c)

            if c == 'area':
                d = data[feature + ' a'] * data[feature + ' b'] * np.pi
            else:
                d = data[feature + ' ' + c]

            d = d - np.mean(d) # get rid of DC off set

            f, p = power_spectral_density(d, time_step, frequency_range=freq_range)

            if n_avrg is not None:
                f, p = avrg(f, n=n_avrg), avrg(p, n=n_avrg)

            if normalize:
                p = p / max(p)

            if plot_type == 'log':
                ax.loglog(f, p, linewidth=3)
            elif plot_type == 'lin':
                ax.plot(f, p, linewidth=3)
            elif plot_type == 'semilogy':
                ax.semilogy(f, p, linewidth=3)
            elif plot_type == 'semilogx':
                ax.semilogx(f, p, linewidth=3)

            # if we plot on a preexisting plot, don't add labels
            if fig is not None:
                if len(annotation_dict) > 0:
                    annotate_frequencies(ax, annotation_dict, higher_harm=1)
                #         annotate_frequencies(ax, annotation_dict, higher_harm=1)

            modes[c] = f[np.argmax(p)]

            ax.set_ylabel(c + ' (norm)')
            ax.set_xlim((min(f), max(f)))

            if i == len(coordinates) - 1:
                ax.set_xlabel('frequency (Hz)')

            if return_data:

                # containers for the data
                p_list.append(p)

    if return_data:
        return fig, axes, {'frequencies': f, 'spectra':p_list, 'coordinates':coordinates}
    else:
        return fig, axes


def plot_ellipse_spectra_zoom(data, info, annotation_dict={}, freq_window=1, n_avrg=None, plot_type='lin', normalize=True, axes=None, verbose=False):
    for mode in ['x', 'y', 'z', '2r', 'r']:
        assert mode in annotation_dict

    if normalize is True:
        normalize = 'max_peak'
    elif normalize is False:
        normalize = 'false'

    assert normalize in ['max_peak', 'false', 'std_dev']

    coordinates = [['x', 'y', 'z'], ['r', '2r', 'm']]
    feature = 'ellipse'

    time_step = 1. / info['info']['FrameRate']
    axes_shape = np.shape(coordinates)


    if axes is None:
        fig, axes = plt.subplots(axes_shape[0], axes_shape[1], sharey=False, sharex=False,
                                 figsize=(8 * axes_shape[1], 3 * axes_shape[0]))
    else:
        fig = None

    if verbose:
        print('axes shape', axes_shape, np.shape(axes), np.shape(coordinates))

    modes = {}
    for i, row in enumerate(zip(coordinates, axes)):

        c_row, a_row = row
        if verbose:
            print('row', i, c_row)
        if verbose:
            print('columns', np.shape(c_row), np.shape(a_row))
        for mode, ax in zip(c_row, a_row):
            if verbose:
                print('====', mode)
            axis_peak = mode
            # if mode in ['x', 'y', 'z']:
            #     data_labels = ['x', 'y', 'area']
            # elif mode in ['r', '2r', 'm']:
            #     data_labels = ['area', 'angle']

            if mode in ['x', 'y']:
                data_labels = [mode]
            elif mode in ['r', '2r', 'm', 'z']:
                data_labels = ['area']

            for data_label in data_labels:

                # as a proxy for the z mode we use the area
                if data_label == 'area':
                    d = data[feature + ' a'] * data[feature + ' b'] * np.pi
                else:
                    d = data[feature + ' ' + data_label]

                d = d - np.mean(d)  # get rid of DC off set


                if normalize == 'std_dev':
                    d /= np.std(d)

                # set the frequency range to be +- freq_window/2 around the peak
                frequency_range = (max(0, annotation_dict[axis_peak][0] - 0.5 * freq_window),
                                   min(annotation_dict[axis_peak][0] + 0.5 * freq_window, 0.5 * info['info']['FrameRate']))

                f, p = power_spectral_density(d, time_step, frequency_range=frequency_range)

                if n_avrg is not None:
                    f, p = avrg(f, n=n_avrg), avrg(p, n=n_avrg)

                if normalize == 'max_peak':
                    p = p / max(p)

                if plot_type == 'log':
                    ax.loglog(f, p, linewidth=3)
                elif plot_type == 'lin':
                    ax.plot(f, p, linewidth=3)
                elif plot_type == 'semilogy':
                    ax.semilogy(f, p, linewidth=3)
                elif plot_type == 'semilogx':
                    ax.semilogx(f, p, linewidth=3)

            # line_length = 0.2
            # # trying to get better lengths for log plots but doesn't work...
            # # if plot_type in ['lin', 'semilogx']:
            # #     line_length = 0.2
            # #
            # # elif plot_type in ['log', 'semilogy']:
            # #     line_length = 1-0.8*min(p / max(p))
            #
            # if len(annotation_dict) > 0:
            #     annotate_frequencies(ax, annotation_dict, x_off=0.1*freq_window, line_length=line_length, higher_harm=1)
            # #         annotate_frequencies(ax, annotation_dict, higher_harm=1)

            modes[mode] = f[np.argmax(p)]
            if normalize == 'max_peak':
                ax.set_ylabel(mode + ' (norm max peak)')
            elif normalize == 'std_dev':
                ax.set_ylabel(mode + ' (norm std)')
            else:
                ax.set_ylabel(mode)
            ax.set_xlim((min(f), max(f)))

            if i == len(coordinates) - 1:
                ax.set_xlabel('frequency (Hz)')

    return fig, axes


def plot_rotations_vs_time(data, time_step, zoom_frames=None, n_avrg=20, axes=None, verbose=False):
    """
    plots a zoom in of the unwrapped angle, the full time trace and the distribution histogram of frequencies
    which are calculated from the derivative of the phase
    Args:
        data:
        time_step:
        zoom_frames:
        n_avrg:
        axes:
        verbose:

    Returns:

    """
    rot_angle = np.unwrap(data['ellipse angle'], discont=get_wrap_angle(data['ellipse angle']))
    if zoom_frames is None:
        min_frame, max_frame = 0, 1000
    else:
        min_frame, max_frame = zoom_frames

    frames = np.arange(min_frame, max_frame)
    t = time_step * frames
    t2 = time_step * np.arange(len(data))

    rot_angle_avrg = avrg(rot_angle, n=n_avrg)
    freqs = np.diff(rot_angle_avrg) / (360 * time_step * n_avrg)

    freq_estimate = np.mean(freqs)

    if axes is None:
        fig, axes = plt.subplots(1, 3, sharey=False, sharex=False, figsize=(18, 4))
    else:
        fig = None

    ax1, ax2, ax3 = axes
    ax1.plot(t, rot_angle[frames] / 360)
    ax2.plot(t2, 1e-3 * rot_angle / 360, label='{:0.2f} Hz'.format(freq_estimate))

    x = ax3.hist(freqs, log=True, bins=50, density=True, alpha=0.3)
    #     m, s = np.mean(rot_freq), np.std(rot_freq)
    #     plt.plot(a, rv.pdf(a), lw = 3)
    #     rv = norm(loc = m, scale = s)
    #     plt.ylim([1e-6, 5e-2])

    if fig is None:
        ax1.set_title('zoom')
        ax2.set_title('full')

        ax1.set_ylabel('rotations')
        ax2.set_ylabel('rotations (x 1000)')
        ax3.set_ylabel('probability density')
        ax1.set_xlabel('time (s)')
        ax2.set_xlabel('time (s)')
        ax3.set_xlabel('rotation freq. (Hz)')

    return fig, axes

def plot_psd_vs_time(x, time_step, start_frame=0, window_length=1000, end_frame=None, full_spectrum=True,
                     frequency_range=None, ax=None, plot_avrg=False, verbose=False, return_data=False):
    """

    Args:
        x: time trace
        time_step: time_step between datapoints
        start_frame: starting frame for analysis (default 0)
        window_length: length of window over which we compute the psd (default 1000)
        end_frame: end frame for analysis (optional if None end_frame is len of total timetrace)
        full_spectrum: if true show full spectrum if false just mark the frequency range
        frequency_range: a tupple or list of two elements frange =[mode_f_min, mode_f_min] that marks a freq range on the plot if full_spectrum is False otherwise plot only the spectrum within the frequency_range
        plot_avrg: if true plot the time averaged PSD on top of the 2D plot

    Returns:

    """

    N_frames = len(x) # total number of frames

    if end_frame is None:
        end_frame = N_frames


    N_windows = (end_frame-start_frame)/window_length # number of time
    N_windows = int(np.floor(N_windows))

    if verbose:
        print('total number of frames:\t\t{:d}'.format(N_frames))
        print('total number of windows:\t{:d}'.format(N_windows))

    # substract mean to get rid of large 0-frequency peak
    x = x-np.mean(x)
    # reshape the timetrace such that each row is a window
    X = x[start_frame:start_frame+window_length*N_windows].values.reshape(N_windows, window_length)
    P = []
    # c = 0
    for x in X:
        # c+=1
        if full_spectrum:
            f, p = power_spectral_density(x, time_step, frequency_range=None)
        else:
            f, p = power_spectral_density(x, time_step, frequency_range=frequency_range)
        P.append(p)
        # print(c,len(p), np.min(p))


    time = np.arange(N_windows) * time_step * window_length

    xlim = [min(f), max(f)]
    ylim = [min(time), max(time)]

    if ax is None:
        if plot_avrg:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax_id = 1
        else:
            fig, ax = plt.subplots(1, 1)
            ax =[ax]
            ax_id = 0

    ax[ax_id].pcolormesh(f, time, np.log(P))

    # print(np.min(P, axis=1), len(np.min(P, axis=1)))

    if not frequency_range is None:
        [mode_f_min, mode_f_max] = frequency_range
        ax[ax_id].plot([mode_f_min, mode_f_min], ylim, 'k--')
        ax[ax_id].plot([mode_f_max, mode_f_max], ylim, 'k--')


    if plot_avrg:
        pmean = np.mean(P, axis=0)
        ax[0].semilogy(f, pmean)
        ax[0].set_ylim([min(pmean), max(pmean)])
        ax[ax_id].set_ylim([min(pmean), max(pmean)])

    ax[ax_id].set_xlim(xlim)
    ax[ax_id].set_ylim(ylim)
    ax[ax_id].set_xlabel('frequency (Hz)')
    ax[ax_id].set_ylabel('time (s)')

    if return_data:
        return fig, ax, {'spectra':P, 'frequencies':f}
    else:
        return fig, ax

def plot_psds(x, time_step, window_ids = None, start_frame = 0, window_length= 1000, end_frame = None,full_spectrum = True, frequency_range= None, ax = None,  plot_avrg = False, return_data=False):
    """

    time trace x is chopped up into segments of length window_length

    for each window we calculate the psd

    Plots all the PSD calculated from the timetrace as a 1D plot
    Args:
        x: time trace
        time_step: time_step between datapoints
        window_ids: the id of the window to be plotted if None, calculate the spectrum from the entire timetrace
        start_frame: starting frame for analysis (default 0)
        window_length: length of window over which we compute the psd (default 1000)
        end_frame: end frame for analysis (optional if None end_frame is len of total timetrace)
        full_spectrum: if true show full spectrum if false just mark the frequency range
        frequency_range: a tupple or list of two elements frange =[mode_f_min, mode_f_min] that marks a freq range on the plot if full_spectrum is False otherwise plot only the spectrum within the frequency_range
        plot_avrg: if true plot the time averaged PSD, windows ids should be None
    Returns:

    """

    if plot_avrg:
        assert window_ids is None

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if window_ids is None:
        if full_spectrum:
            f, p = power_spectral_density(x, time_step, frequency_range=None)
        else:
            f, p = power_spectral_density(x, time_step, frequency_range=frequency_range)
        ylim = [min(p), max(p)]
    else:

        N_frames = len(x) # total number of frames

        if end_frame is None:
            end_frame = N_frames


        N_windows = (end_frame-start_frame)/window_length # number of windows
        N_windows = int(np.floor(N_windows))

        print('total number of frames:\t\t{:d}'.format(N_frames))
        print('total number of windows:\t{:d}'.format(N_windows))

        # reshape the timetrace such that each row is a window
        X = x[start_frame:start_frame+window_length*N_windows].reshape(N_windows, window_length)
        P = []
        if plot_avrg:

            for x in X:
                if full_spectrum:
                    f, p = power_spectral_density(x, time_step, frequency_range=None)
                else:
                    f, p = power_spectral_density(x, time_step, frequency_range=frequency_range)
                P.append(p)


            pmean = np.mean(P, axis=0)
            ax.semilogy(f, pmean)
            ax.set_ylim([min(pmean), max(pmean)])
        else:
            for id, x in enumerate(X):

                if id in window_ids:

                    if full_spectrum:
                        f, p = power_spectral_density(x, time_step, frequency_range=None)
                    else:
                        f, p = power_spectral_density(x, time_step, frequency_range=frequency_range)
                    P.append(p)
                    ylim = [np.min(P), np.max(P)]



    xlim = [min(f), max(f)]


    if window_ids is None:
        ax.semilogy(f, p, '-')
    else:
        for p, id in zip(P, window_ids):
            ax.semilogy(f, p, '-', label=id)
        plt.legend(loc=(1, 0))

    if not frequency_range is None:
        [mode_f_min, mode_f_max] = frequency_range
        ax.plot([mode_f_min, mode_f_min], ylim, 'k--')
        ax.plot([mode_f_max, mode_f_max], ylim, 'k--')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('psd (arb.u.)')

    if return_data:
        return fig, ax, {'spectra': P, 'frequencies': f}
    else:
        return fig, ax

def plot_timetrace(x, time_step, window_length =1, start=None, end =None, start_end_unit = 'frames', ax = None, verbose = False):
    """
    Takes a file or sequential files with a set of bead positions (such as created by extract_motion), and computes and plots the ringdown
    filepaths: a list of filepaths to .csv files containing the bead position trace in (x,y) coordinates
    frequency: oscillation frequency of mode
    window_width: width of window centered on frequency over which to integrate
    fps: frame rate in frames per second of the data
    bead_diameter: diameter (in um) of the bead
    axis: either 'x' or 'y', specifies which direction to look at oscillations in (if using the reflection on
        the right wall, this is x for z mode, y for xy modes)
    starting_frame: all frames before this one will not be included
    save_filename: if provided, uses this path to save the plot, otherwise saves in the original filepath .png
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    time = np.arange(len(x)) * time_step * window_length
    ax.plot(time, x)
    ax.set_xlabel('time (s)')

    return fig, ax

def plot_tracking_error(data, methods):


    y_pos = np.arange(len(methods))
    for channel in ['x', 'y']:
        errors = np.array([tracking_error(data['xo'], data['{:s} {:s}'.format(channel, m)]) for m in methods])
        sign = -1 if channel == 'x' else +1
        for i, err_type in zip([0,1], ['', 'diff']):
            # plt.figure()
            plt.bar(y_pos+sign*((i+0.5)*0.2), errors[:,i], align='center', alpha=0.5, width = 0.2, label = '{:s} {:s}'.format(channel, err_type))
            plt.xticks(y_pos, methods)

    plt.title('Tracking error ({:s})'.format(channel))
    plt.ylabel('Error')
    plt.legend(loc = (1,0.5))

# OLD STUFF!!!!
def plot_video_frame_old_stuff(file_path, frames, xy_position = None, gaussian_filter_width=None, xylim = None, roi = None, ax = None, radius = 3):
    """

    plots frames of the video

    Args:
        file_path: path to video file
        frames: integer or list of integers of the frames to be plotted
        xy_position: xy position of the magnet. If provided, this position will be plotted on top of  the image
        gaussian_filter_width: if not None apply Gaussian filter
        xylim: xylim to zoom in on a region, if not specified (ie None) show full image

        roi: region of interest, this allows to limit the search to a region of interest with the frame, the structure is
        roi = [roi_center, roi_dimension], where
            roi_center = [ro, co], roi_dimension = [h, w], where
            ro, co is the center of the roi (row, columns) and
            w, h is the width and the height of the roi

            Note that roi dimensions w, h should be odd numbers!

        radius: sets the radiusof the circle that indicte the position xy

    """


    if not hasattr(frames, '__len__'):
        frames = [frames]

    if ax is None:
        fig, ax = plt.subplots(1, len(frames))
        # if frames is an array of len=1, the ax object is not a list, so for the following code we make it into a list
        if len(frames)==1:
            ax =[ax]
    else:
        fig = None


    if not roi is None:
        [roi_center, roi_dimension] = roi

    v = pims.Video(file_path)
    video = rgb2gray_pipeline(v)

    if not gaussian_filter_width is None:
        gaussian_filter_pipeline = pipeline(gaussian_filter)
        video = gaussian_filter_pipeline(video, gaussian_filter_width)


    frame_shape = np.shape(video[frames[0]])


    for frame, axo in zip(frames, ax):

        image = video[frame]


        axo.imshow(image, cmap='pink')
        # axo.imshow(video[frame], cmap='pink')
        if not xy_position is None:



            # note that we flip the x and y axis, because that is how
            # axo.plot(xy_position[frame, 1], xy_position[frame, 0], 'xg', markersize = 30, linewidth = 4)
            circ = Circle((xy_position[frame, 1], xy_position[frame, 0]), radius =radius, linewidth=2, edgecolor='g', facecolor='none')
            axo.add_patch(circ)

            # plot also the positions obtained with center-of-mass
            if len ( xy_position[frame]) == 4:
                circ = Circle((xy_position[frame, 3], xy_position[frame, 2]), radius=radius, linewidth=2, edgecolor='r',
                              facecolor='none')
                axo.add_patch(circ)
            # plot also the positions obtained with trackpy
            if len ( xy_position[frame]) == 6:
                # axo.plot(xy_position[frame, 3], xy_position[frame, 2], 'xr', markersize=30, linewidth = 2)
                # the postions in trackpy are the usual x,y order
                circ = Circle((xy_position[frame, 5], xy_position[frame, 4]), radius=radius, linewidth=2, edgecolor='r',
                              facecolor='none')
                axo.add_patch(circ)
        if xylim is None:
            xlim = [0, frame_shape[0]]
            ylim = [0, frame_shape[1]]
        else:
            xlim, ylim = xylim

        axo.set_xlim(xlim)
        axo.set_ylim(ylim)
        # plt.show()


        # Create a Rectangle patch to show roi
        if not roi is None:
            rect = Rectangle((int(roi_center[1] - roi_dimension[1] / 2)-1, int(roi_center[0] - roi_dimension[0] / 2)), roi_dimension[1], roi_dimension[0], linewidth=1, edgecolor='r', facecolor='none')
            axo.add_patch(rect)

    return fig, ax




if __name__ == '__main__':
    pass