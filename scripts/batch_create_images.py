import os
from track_sphere.plot_data import plot_ellipse_spectra, plot_ellipse_spectra_zoom
from track_sphere.read_write import load_time_trace

from glob import glob

import matplotlib.pyplot as plt

image_folder = '../images/'
source_folder_positions = '../processed_data/20180607_Sample_6_bead_1/position_data/'

annotation_dict = {'2r': [13.16877399924, 0.6],
                   'm': [544.131481993, 0.6],
                   'r': [6.58438699962, 0.85],
                   'x': [171.52045677, 0.85],
                   'y': [177.690573472, 0.6],
                   'z': [402.859033756, 0.6]}

freq_window = 0.5
method = 'fit_ellipse'
plot_type='lin'


position_file_names = sorted([os.path.basename(f) for f in glob(source_folder_positions + '*-'+method+'.dat')])

for filename in position_file_names[0:2]:
    print(filename)
    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    fig = plot_ellipse_spectra_zoom(data, info, annotation_dict=annotation_dict, freq_window=freq_window, n_avrg=None, plot_type = plot_type, normalize=True)
    fig.suptitle('mode spectra ' + info['filename'], fontsize=20)
    print(os.path.join(image_folder, info['filename'].replace('.avi', '-' + method + '-modes-' + plot_type + '.jpg')))
    # fig.savefig(os.path.join(image_folder, info['filename'].replace('.avi', '-' + method + '-modes-' + plot_type + '.jpg')))
    fig.savefig('test.jpg')

    plt.show()
