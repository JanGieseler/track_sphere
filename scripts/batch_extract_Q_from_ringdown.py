
from track_sphere.plot_data import plot_timetrace_energy, plot_fit_exp_decay, plot_psd_vs_time
from track_sphere.utils import get_position_file_names, power_to_energy_K
from track_sphere.read_write import load_info, load_time_trace
import matplotlib.pyplot as plt
import os

# settings - run 73
run = 73
mode = 'x'
window_length = 5000
frequency_window = 20
source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
method = 'fit_ellipse'
magnet_diameter = 45
image_folder = '../images/20180628_Sample_6_Bead_1/ring-down/'

################################################################################
#### run the script
################################################################################
filename = get_position_file_names(source_folder_positions, method=method, runs=[run])[0]

print(filename)

data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)

x_pos = data['ellipse '+mode]

calib = load_info(filename=filename, folder_positions=source_folder_positions)['ellipse']['calibration factor: (um/px)']
fo = load_info(filename=filename, folder_positions=source_folder_positions)['ellipse'][mode]
fs = load_info(filename=filename, folder_positions=source_folder_positions)['info']['FrameRate']
dt = 1 / fs

if fo > fs / 2:
    frequency_range = (fs - fo - frequency_window / 2, fs - fo + frequency_window / 2)
else:
    frequency_range = (fo - frequency_window / 2, fo + frequency_window / 2)

fig, ax, (t, x_energy, f) = plot_timetrace_energy(x=x_pos, time_step=dt, window_length=window_length,
                                                  frequency_range=frequency_range, return_data=True)

x_energy = power_to_energy_K(x_energy, radius=magnet_diameter / 2, frequency=fo, calibration_factor=calib, density=7600)
fig, ax, fit = plot_fit_exp_decay(t, x_energy, t_min=1.1e4 * dt, t_max=500, return_data=True)
ax.set_ylabel('energy (Kelvin)')

image_filename = os.path.join(image_folder, filename.replace('-fit_ellipse.dat', '-ring-down.jpg'))
fig.savefig(image_filename)
plt.close(fig)

fig, ax, psd_data = plot_psd_vs_time(x=x_pos, time_step=dt, frequency_range=frequency_range,
                                     full_spectrum=True, return_data=True)

image_filename = os.path.join(image_folder, filename.replace('-fit_ellipse.dat', '-spectogram.jpg'))
fig.savefig(image_filename)
plt.close(fig)

