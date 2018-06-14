import os
import yaml
import pandas as pd
import cv2 as cv


def load_video_info(filename):
    """
    loads the video metadata that has been exported with metainfo (www.MediaArea.net) into a json file

    Args:
        filename: path to the .avi or .xml file

    Returns: ['FileSize', 'FrameRate', 'BitDepth', 'Width', 'Duration', 'FrameCount', 'Height', 'CodecID']

    """
    if len(filename.split('.avi')) == 2:
        filename = filename.replace('.avi', '.json')

    print(os.path.normpath(filename))
    assert os.path.exists(filename)

    with open(filename, 'r') as infile:
        data = yaml.safe_load(infile)

    data = data['media']['track']
    # select the relevant paramters
    info = {key: data[0][key] for key in ['FrameRate', 'FileSize', 'Duration']}
    info.update({key: data[1][key] for key in ['Width', 'Height', 'BitDepth', 'CodecID', 'FrameCount']})

    # now convert to numbers
    info.update({key: int(info[key]) for key in ['FrameCount', 'BitDepth', 'FileSize', 'Width', 'Height']})
    info.update({key: float(info[key]) for key in ['FrameRate', 'Duration']})
    info['filename'] = filename

    return info


def load_time_trace(filename, source_folder_positions=None, methods=[], verbose=False):
    """
    Takes in the located in source_folder_positions
    from which containing the bead positions

    Args:
        source_folder_positions: folderpath positions data (.dat files obtained with track_sphere.extract_position_data)
        filename: name of video file from which position data was extracted or .dat file with the position information
        methods: list of methods used to extract position data

        verbose: if True print some output

    Returns: the position data as a pandas dataframe

    """

    # if no source folder is provided take the directory name from the video_filename
    if source_folder_positions is None:
        source_folder_positions = os.path.dirname(filename)

    # if we pass in the video file
    if filename[-4:] == '.avi':
        filepath = os.path.join(source_folder_positions, filename)
    elif filename[-4:] == '.dat':
        assert methods is not [] # is we pass in a .dat file, we get the method from the filename

        methods = [filename.strip('.dat').split('-')[-1]] # method is added to video name as -method.dat

        if verbose:
            print('loading methods', methods)

        # in case we get the dat file we create a filepath for the video_file (it might not exist) from which we can later
        # reconstruct the name of the json file with the metadata
        filepath = os.path.join(source_folder_positions, filename.split('-' + methods[0])[0]) + '.avi'

    assert len(methods)>0




    if methods is []:
        print('define method!')

    info = {'filename': os.path.basename(filepath)}
    data = None
    for m, method in enumerate(methods):

        with open(filepath.replace('.avi', '-{:s}.json'.format(method)), 'r') as infile:
            info_m = yaml.safe_load(infile)

        if m == 0:
            # load data
            data = pd.read_csv(filepath.replace('.avi', '-{:s}.dat'.format(method)), index_col=0)
            info['info'] = info_m['info']
        else:
            data = data.join(pd.read_csv(filepath.replace('.avi', '-{:s}.dat'.format(method)), index_col=0))
        info[method] = {key: info_m[key] for key in ['method', 'skipped_frames', 'method_parameters']}

        if verbose:
            print('{:s} skipped frames: {:d}'.format(method, len(info[method]['skipped_frames'])))

    if verbose:
        print('data set contains: ', data.keys())
        print('data shape is: ', data.shape)

    return data, info


def grab_frame(file_in, frame_id=0, verbose=False):
    """

    opens video can with opencv and  returns frame

    Args:
        file_in: name of video file
        frame_id: id of frame to be loaded
        verbose: if true send print statements to console

    Returns:

    """
    cap = cv.VideoCapture(file_in, False)  # open input video

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)  # set the starting frame for reading to min frame
    ret, frame_in = cap.read()

    cap.release()

    if verbose:
        print(file_in, ':', ret)

    return frame_in
