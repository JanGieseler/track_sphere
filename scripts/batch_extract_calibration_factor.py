
from track_sphere.read_write import load_time_trace, update_info
from track_sphere.utils import get_calibration_factor, get_position_file_names

source_folder_positions = '../processed_data/20180628_Sample_6_bead_1/position_data/'
method = 'fit_ellipse'
particle_diameter=45  # um

# get all the files and sort them by the run number
position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(60, 180)))



################################################################################
#### run the script
################################################################################
for i, filename in enumerate(position_file_names):

    print(filename)


    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    return_dict = get_calibration_factor(data, particle_diameter=particle_diameter, verbose=False)

    update_info(filename, key='calibration factor: (um/px)', value=return_dict['calibration factor: (um/px)'],
                folder_positions=source_folder_positions, dataset='ellipse', verbose=True)
