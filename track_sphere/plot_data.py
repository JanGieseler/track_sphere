import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Rectangle, Circle

import numpy as np
from track_sphere.read_write import grab_frame

from track_sphere.utils import power_spectral_density, avrg, get_wrap_angle, get_rotation_frequency, fit_exp_decay
from track_sphere.utils import exp_offset, power_to_energy_K
from track_sphere.read_write import load_time_trace
from track_sphere.plot_utils import annotate_frequencies

def plot_ellipse_spectra(data, info, annotation_dict={}, freq_range=None, n_avrg=None, plot_type = 'lin', verbose=False, normalize=True, return_data=False, axes = None):
    """

    Args:
        data:
        info:
        annotation_dict:
        freq_range:
        n_avrg: average n_avrg successive datapoint to smoothen the data
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
    for mode in ['x', 'y', 'z', '2r', 'r', 'm']:
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


def plot_rotations_vs_time(data, time_step, t_max=0.1, n_avrg=20,n_avrg_unwrapped=20, axes=None, verbose=False, wrap_angle=None):
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
        wrap_angle: if None find wrap_angle automatically otherwise wrap angles at this value

    Returns:

    """


    if wrap_angle is None:
        wrap_angle = get_wrap_angle(data['ellipse angle'], navrg=n_avrg)

    if verbose:
        print('wrap_angle', wrap_angle)


    min_frame, max_frame = 0, int(t_max/time_step/n_avrg)


    if axes is None:
        fig, axes = plt.subplots(2, 2, sharey=False, sharex=False, figsize=(18, 8))
    else:
        fig = None

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]




    rot_angle = data['ellipse angle']



    rot_angle = avrg(rot_angle, n=n_avrg)
    x = ax4.hist(rot_angle, log=True, bins=50, alpha=0.3)

    rot_angle = np.unwrap(rot_angle, discont=wrap_angle)

    frames = np.arange(min_frame, max_frame)
    t = time_step * frames* n_avrg
    ax1.plot(t, rot_angle[frames] / 360)

    t2 = time_step *n_avrg* np.arange(len(rot_angle))
    if max(rot_angle)>1e3:
        ax2.plot(t2, 1e-3 * rot_angle / 360)
        label2 = 'rotations (x 1000)'
    else:
        ax2.plot(t2, rot_angle / 360)
        label2 = 'rotations'


    rot_angle = avrg(rot_angle, n=n_avrg_unwrapped)
    freqs = np.diff(rot_angle) / (360 * time_step * n_avrg_unwrapped)

    freq_estimate = np.mean(freqs)

    if verbose:
        print('{:d} datapoints'.format(len(rot_angle)))
        print('freq_estimate {:0.3f}'.format(freq_estimate))
        print('wrap angle {:0.2f} deg'.format(wrap_angle))



    x = ax3.hist(freqs, log=True, bins=50, alpha=0.3)


    # if fig is None:
    ax1.set_title('zoom')
    ax2.set_title('full')

    ax1.set_ylabel('rotations')
    ax2.set_ylabel(label2)
    ax3.set_ylabel('probability density')
    ax1.set_xlabel('time (s)')
    ax2.set_xlabel('time (s)')
    ax3.set_xlabel('rotation freq. (Hz)')

    ax4.set_xlabel('angles')
    ax4.set_ylabel('prob density')

    return fig, axes

def plot_psd_vs_time(x, time_step, start_frame=0, window_length=1000, end_frame=None, full_spectrum=True,
                     frequency_range=None, ax=None, plot_avrg=False, verbose=False, return_data=False, plot_hist=False):
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
        plot_hist: if true plot the hisogram of the max freq per window on top of the 2Dplot

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
        elif plot_hist:
            fig, ax = plt.subplots(1, 2)
            ax_id = 0
        else:
            fig, ax = plt.subplots(1, 1)
            ax =[ax]
            ax_id = 0

    if plot_hist:
        ax[ax_id].pcolormesh(f, time, P)
    else:
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


    elif plot_hist:
        fmax = f[np.argmax(P, axis=1)]
        ax[1].hist(fmax)
        # ax[0].set_ylim([min(pmean), max(pmean)])
        # ax[ax_id].set_ylim([min(pmean), max(pmean)])


    ax[ax_id].set_xlim(xlim)
    ax[ax_id].set_ylim(ylim)
    ax[ax_id].set_xlabel('frequency (Hz)')
    ax[ax_id].set_ylabel('time (s)')

    if return_data:
        return fig, ax, {'spectra': P, 'frequencies': f}
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
    else:
        fig = None

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

def plot_timetrace_energy(x, time_step, window_length =1, start_frame=0, end_frame=None, frequency_range= None, ax = None, verbose = False, return_data=False):
    """

    Args:
        x: position time trace
        time_step:
        window_length: integration window for calculation of the energy should be much larger than typical freq and much smaller than decay time
        start_frame:
        end_frame:
        frequency_range:
        ax:
        verbose:
        return_data:

    Returns:

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    N_frames = len(x) # total number of frames

    if end_frame is None:
        end_frame = N_frames



    N_windows = (end_frame-start_frame)/window_length # number of windows
    N_windows = int(np.floor(N_windows))

    if verbose:
        print('total number of frames:\t\t{:d}'.format(N_frames))
        print('total number of windows:\t{:d}'.format(N_windows))

    # reshape the timetrace such that each row is a window
    X = x[start_frame:start_frame+window_length*N_windows].reshape(N_windows, window_length)
    P = []
    F = []
    for id, x in enumerate(X):
        f, p = power_spectral_density(x, time_step, frequency_range=frequency_range)
        P.append(p)
        F.append(f[np.argmax(p)])

    df = np.mean(np.diff(f))

    # now calculate the energy (P is in units of px^2/Hz or m^2/Hz)
    x = np.sum(P, axis=1)*df

    time = np.arange(len(x)) * time_step * window_length
    ax.plot(time, x)
    ax.set_xlabel('time (s)')

    if return_data:
        return fig, ax, (time, x, F)
    else:
        return fig, ax


def plot_fit_exp_decay(t, x, t_min=0, t_max=None, return_data=False, axes=None, verbose=False):
    """
    plots the energy x over time between t_min and t_max and fits to an exponential decay

    Args:
        t:
        x:
        t_min:
        t_max:
        return_data:
        axes:

    Returns:

    """
    x2 = x[t > t_min]
    t2 = t[t > t_min]

    if t_max is not None:
        x2 = x2[t2 < t_max]
        t2 = t2[t2 < t_max]

    if axes is None:
        fig, axes = plt.subplots(1, 1)
    else:
        fig = None

    fit = fit_exp_decay(t, x, offset=True, verbose=verbose)

    axes.plot(t2, x2, 'o')
    axes.plot(t2, exp_offset(t2, *fit[0]))
    axes.set_xlabel('time (s)')
    axes.set_ylabel('energy')

    if return_data:
        return fig, axes, fit
    else:
        return fig, axes

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


def plot_rot_angle_dist(data, info, frame_max=100, n_avrg_list = [1,2, 4, 8, 16, 32, 64, 128]):
    """
    plots the distribution of angles as a function of binning length
    Args:
        data:
        time_step:
        source_folder_positions:
        verbose:
        frame_max:

    Returns:

    """
    time_step = info['info']['FrameRate']
    rot_angle = data['ellipse angle']

    fig, axes = plt.subplots(2, 2, sharey=False, sharex=False, figsize=(20, 8))
    res = []

    for n_avrg in n_avrg_list:

        rot_angle_avrg = avrg(rot_angle, n=n_avrg)

        x = axes[0,0].hist(rot_angle_avrg, log=True, bins=50, density=True, alpha=0.3)

        res.append(get_rotation_frequency(data, info)[0:2])

        axes[1,0].plot(np.arange(int(frame_max/n_avrg))*time_step*n_avrg, rot_angle_avrg[:int(frame_max/n_avrg)], '.-', label='n_avrg={:d}'.format(n_avrg))

    axes[0,1].semilogx(n_avrg_list, [abs(val[0]) for val in res],'o')
    axes[0,1].set_xlabel('number of averages')
    axes[0,1].set_ylabel('mean freq')

    axes[1,1].loglog(n_avrg_list, [val[1] for val in res], 'o')
    axes[1,1].set_xlabel('number of averages')
    axes[1,1].set_ylabel('std dev freq')


    axes[0,0].set_xlabel('angle (deg')
    axes[0,0].set_ylabel('prob. density')

    axes[1,0].set_xlabel('time (s)')
    axes[1,0].set_ylabel('angle (deg)')

    return fig, axes


def waterfall(position_file_names,source_folder_positions=None, modes='xy', navrg=10, off_set_factor=-3, xlim=None, tag='_Sample_6_Bead_1_', nmax=None,
              method = 'fit_ellipse', verbose=False):
    """

    calculated the psds and plots them as a waterfall plot

    Args:
        position_file_names:
        modes:
        navrg:
        off_set_factor:
        xlim:
        tag:
        nmax:
        verbose:

    Returns:

    """
    fig, ax = plt.subplots(len(modes), 1, sharex=True, figsize=(18, 5 * len(modes)))

    for i, filename in enumerate(position_file_names):
        run = int(filename.split(tag)[1].split('-')[0])
        if verbose:
            print(filename, run)
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
        dt = 1./info['info']['FrameRate']
        # calculate the psd
        psd_data = {}
        for mode in modes:
            if method == 'fit_ellipse':
                if mode == 'r':
                    x = data['ellipse angle']
                elif mode == 'z':
                    x = data['ellipse a'] * data['ellipse b']
                else:
                    x = data['ellipse ' + mode]
            elif method.lower() == 'bright px':
                x = data['bright px ' + mode]

            x -= np.mean(x)
            if nmax is not None:
                x = x[0:nmax]
            f, p = power_spectral_density(x, time_step=dt)
            psd_data[mode] = p
        psd_data['f'] = f

        f = psd_data['f'][1:]  # get rid of first point (DC)
        f = avrg(f, navrg)  # refold (assume that signal is aliasied)
        #         f = np.mean(df_modes['FrameRate'])-avrg(f, navrg)  # refold (assume that signal is aliasied)

        for a, mode in enumerate(modes):
            d = psd_data[mode]  # get data
            d = avrg(d[1:] * 10 ** (off_set_factor * i), navrg)  # shift on a log scale to get the waterfall effect
            ax[a].semilogy(f, d, label=str(run), alpha=0.5)  # plot

    if xlim is not None:
        assert len(xlim) == 2
        ax[0].set_xlim(xlim)
    for a, mode in enumerate(modes):
        ax[a].set_title(mode + ' data')
    ax[a].set_xlabel('frequency (Hz)')
    plt.legend(loc=(1, 0.0))

    return fig


def plot_get_ring_down_time(x, time_step,frequency_range, window_length,
                 t_min=0, t_max=None, fo=None,  calib=1, magnet_diameter=1, density=7600,
                 mode='', return_fig=False):


    # if fo is not provided set it to the center of the range
    if fo is None:
        fo = np.mean(frequency_range)
    if t_max is None:
        t_max = len(x)*time_step

    fig, ax, (t, x_energy, f) = plot_timetrace_energy(x=x, time_step=time_step, window_length=window_length,
                                                      frequency_range=frequency_range, return_data=True)
    plt.close(fig)

    x_energy = power_to_energy_K(x_energy, radius=magnet_diameter / 2, frequency=fo, calibration_factor=calib,
                                 density=density)
    fig1, ax, fit = plot_fit_exp_decay(t, x_energy, t_min=t_min, t_max=t_max, return_data=True)


    # image_filename = os.path.join(image_folder, filename.replace('-fit_ellipse.dat', '-ring-down.jpg'))
    # image_filename = os.path.join(image_folder, filename.replace('-fit_ellipse.dat', '-spectogram.jpg'))


    if return_fig:

        ax.set_ylabel('energy (Kelvin)')

        ax.set_title('Q = {:0.0f}, f_{:s} = {:0.0f} Hz'.format(2 * np.pi * fo * fit[0][1], mode, fo))

        fig2, ax, psd_data = plot_psd_vs_time(x=x, time_step=time_step, frequency_range=frequency_range,
                                              window_length=window_length,
                                              full_spectrum=False, return_data=True)


        return fit[0], (fig1, fig2)
    else:
        plt.close(fig1)

        return fit[0]




if __name__ == '__main__':
    pass