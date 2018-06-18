import os
import yaml
import json
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
    info = {key: data[0][key] for key in ['FrameRate', 'FileSize', 'Duration', 'File_Modified_Date', 'File_Modified_Date_Local']}
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

    assert len(methods) > 0

    if methods is []:
        print('define method!')

    info = {'filename': os.path.basename(filepath)}
    data = None
    for m, method in enumerate(methods):

        with open(filepath.replace('.avi', '-{:s}.json'.format(method)), 'r') as infile:
            info_m = yaml.safe_load(infile)

        if m == 0:
            # load data
            # index_col=0 produces a warning not quite clear why
            # check: https://stackoverflow.com/questions/48818335/why-pandas-read-csv-issues-this-warning-elementwise-comparison-failed
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

def load_info(filename, folder_positions=None, verbose=False, return_filname=False):
    """
    loads the info file that is created together with the position data
    Args:
        filename: name of info file (.json) either just filename then need to provide also folder_positions or full filepath
        folder_positions: (optional) folder of where info file is located

    Returns: dictionary with info

    """

    # we want the json file, in case we receive the .dat file
    filename = filename.split('.')[0] + '.json'

    if folder_positions is not None:
        filename = os.path.join(folder_positions, filename)

    #check that file exists
    if verbose and not os.path.exists(filename):
        print('filename', filename)
    assert os.path.exists(filename), filename

    with open(filename, 'r') as infile:
        info = yaml.safe_load(infile)

    if return_filname:
        return info, filename
    else:
        return info


def update_info(filename, key, value, folder_positions=None, dataset='ellipse', verbose=False):
    """
    loads the info file that is created together with the position data and updates or adds values to the 'dataset',
    e.g. the mode frequencies
    Args:
        filename: name of info file (.json) either just filename then need to provide also folder_positions or full filepath
        key: name of the parameter to be updated / added
        value: value of the parameter to be updated / added
        folder_positions: (optional) folder of where info file is located
        dataset: name of dataset where to updat / add value
        verbose: if True print outputs

    Returns: None

    """


    info, filename = load_info(filename, folder_positions=folder_positions, verbose=verbose, return_filname=True)

    if dataset in info:
        if verbose:
            if key in info[dataset]:
                print('updating ' + dataset + '.' + key + ': ' + str(info[dataset][key]) + ' => '+str(value))
            else:
                print('adding ' + dataset + '.' + key + ': ' + str(value))
        info[dataset][key] = value

    else:
        info.update({dataset: {key: value}})

    # write back to file
    # with open(filename.replace('.json','.txt'), 'w') as outfile:
    #     tmp = json.dump(info, outfile, indent=4)
    with open(filename, 'w') as outfile:
        tmp = json.dump(info, outfile, indent=4)

    if verbose:
        print('updated ' + filename)





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
