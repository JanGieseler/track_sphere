# This batch script extracts the rotation angle ellipse position data files and addes the result
# to the .json file that comes with the position data fil

import os
from glob import glob
from track_sphere.read_write import load_time_trace, update_info
from track_sphere.utils import get_rotation_frequency

source_folder_positions = '../processed_data/20180607_Sample_6_bead_1/position_data/'
method = 'fit_ellipse'





################################################################################
#### run the script
################################################################################
# get all the files and sort them by the run number
position_file_names = sorted([os.path.basename(f) for f in glob(source_folder_positions + '*-'+method+'.dat')])
position_file_names = sorted(position_file_names, key=lambda f: int(f.split('-')[0].split('Bead_')[1].split('_')[0]))

axes = None
for i, filename in enumerate(position_file_names[7:16]):
    print(filename)

    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    ret = get_rotation_frequency(data, info, n_avrg=20)

    update_info(filename, 'rotation_freq', {k:v for k, v in zip(['mean', 'std', 'time', 'n_avrg'], ret)}
    , folder_positions=source_folder_positions, dataset='ellipse', verbose=True)

