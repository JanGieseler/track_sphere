import os, yaml
import numpy as np
import cv2 as cv
# import colors
import pandas as pd
import datetime
def load_video_info(filename):
    """
    loads the video metadata that has been exported with metainfo (www.MediaArea.net) into a json file

    Args:
        filename: path to the .avi or .xml file

    Returns: ['FileSize', 'FrameRate', 'BitDepth', 'Width', 'Duration', 'FrameCount', 'Height', 'CodecID']

    """
    if len(filename.split('.avi'))==2:
        filename = filename.replace('.avi', '.json')

    print(os.path.normpath(filename))
    assert os.path.exists(filename)


    with open(filename, 'r') as infile:
        data = yaml.safe_load(infile)

    data = data['media']['track']
    info = {}
    # select the relevant paramters
    info = {key: data[0][key] for key in ['FrameRate', 'FileSize', 'Duration']}
    info.update({key: data[1][key] for key in ['Width', 'Height', 'BitDepth', 'CodecID', 'FrameCount']})

    # now convert to numbers
    info.update({key: int(info[key]) for key in ['FrameCount', 'BitDepth', 'FileSize', 'Width', 'Height']})
    info.update({key: float(info[key]) for key in ['FrameRate', 'Duration']})

    return info

def ffmpeg_reencode_video(filepath, filepath_target = None, start_time=0):
    """

    uses ffmpeg to reencode the video. This is usually
    necessary for the 1200fps videos, which are timestamped weirdly and often have the first second
    corrupted. If you turn this off and receive the error "AVError: [Errno 1094995529] Invalid data found
    when processing input", try turning this on. Will double the runtime of the function.

    Args:
        filepath: path to file of original video
        filepath_target: target file (optional) if None same as input with replacing ".avi" by  "_reencode.avi"
        start_time: start time of video in seconds)

    Returns: nothing but writes reencoded file to disk

    """

    if filepath_target is None:
        filepath_target = filepath.replace('.avi', '_reencode.avi')

    print('start time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    # command string: ffmpeg -i Z:\...\ringdown.avi -s 1 -c copy Z:\...\ringdown_reencode.avi
    # calls ffmpeg, -i specifies input path, -ss 1 cuts first second of video, -c copy copies
    # the input codec and uses it for the output codec, and the last argument is the output file
    # cutting the first second isn't always necessary, but sometimes the videos will not load without it

    if start_time>0:
        cmd_string = "ffmpeg -i " + filepath + ' -ss {:d} -c copy '.format(start_time) + filepath_target


    # performs system (command line) call
    x = os.system(cmd_string)

    print('end time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('wrote:\n{:s}'.format(filepath_target))

def ffmpeg_segment_video(file_in, segmentation_frames=None):
    """

    uses ffmpeg to segment the video

    Args:
        file_in: path to file of original video
        segmentation_frames: list of frames where to cut the video or number that gives the number of segments

    Returns: nothing but writes reencoded file to disk into subfolder with name of video-segmented

    """


    if segmentation_frames is None:
        info = load_video_info(file_in)
        print('no sementation frames provdided. Give list of frames or number of segmentations!')
        print(info)
        return

    elif not isinstance(segmentation_frames, list):
        frame_count = load_video_info(file_in)['FrameCount']
        segmentation_length = int(frame_count/segmentation_frames)
        print('video has {:d} frames, will segment into {:d} frames of length {:d}'.format(
            frame_count,
            segmentation_frames,
            segmentation_length
        ))

        segmentation_frames = list(range(segmentation_length,frame_count-segmentation_length, segmentation_length))

    # turn numbers into a list of strings
    segmentation_frames = ['{:d}'.format(f) for f in segmentation_frames]
    number_of_frame_digits = len(str(len(segmentation_frames)))
    # turn list of strings into single string
    segmentation_frames = ','.join(segmentation_frames)

    file_out = file_in.replace('.avi', '-chop%0'+ str(number_of_frame_digits) + 'd.avi')


    subfolder = os.path.basename(file_in).replace('.avi', '-segmented')
    segmentation_dir = os.path.join(os.path.join(os.path.dirname(file_out), subfolder))

    if not os.path.exists(segmentation_dir):
        os.makedirs(segmentation_dir)

    file_out = os.path.join(segmentation_dir, os.path.basename(file_out))
    # file_out = os.path.join(os.path.dirname(file_out), os.path.basename(file_out))

    file_segment = file_out.replace('-chop%0'+ str(number_of_frame_digits) + 'd.avi', '-segments.csv')

    # Segment the input file by splitting the input file according to the frame numbers sequence
    # specified with the segment_frames option:
    # should be something like:
    # print('ffmpeg -i ../example_data/20171207_magnet.avi -codec copy -map 0 -f segment -segment_list ../example_data/20171207_magnet-segments.csv -segment_frames 100,200,300,500,800 ../example_data/20171207_magnet-chop%03d.avi')
    # cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy -map 0 -f '
    # cmd_string += 'segment -segment_list ' + file_segment + ' -segment_frames '+segmentation_frames+' ' + file_out

    cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy -map 0 -f '
    cmd_string += 'segment -segment_list ' + file_segment + ' -segment_frames ' + segmentation_frames + ' -reset_timestamps 1 ' + file_out



    print('start time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print(cmd_string)
    x = os.system(cmd_string)
    print('end time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('wrote:\n{:s}'.format(file_out))

def ffmpeg_segment_video(file_in, segmentation_frames=None):
    """

    uses ffmpeg to segment the video

    Args:
        file_in: path to file of original video
        segmentation_frames: list of frames where to cut the video or number that gives the number of segments

    Returns: nothing but writes reencoded file to disk into subfolder with name of video-segmented

    """


    if segmentation_frames is None:
        info = load_video_info(file_in)
        print('no sementation frames provdided. Give list of frames or number of segmentations!')
        print(info)
        return

    elif not isinstance(segmentation_frames, list):
        frame_count = load_video_info(file_in)['FrameCount']
        segmentation_length = int(frame_count/segmentation_frames)
        print('video has {:d} frames, will segment into {:d} frames of length {:d}'.format(
            frame_count,
            segmentation_frames,
            segmentation_length
        ))

        segmentation_frames = list(range(segmentation_length,frame_count-segmentation_length, segmentation_length))

    # turn numbers into a list of strings
    segmentation_frames = ['{:d}'.format(f) for f in segmentation_frames]
    number_of_frame_digits = len(str(len(segmentation_frames)))
    # turn list of strings into single string
    segmentation_frames = ','.join(segmentation_frames)

    file_out = file_in.replace('.avi', '-%0'+ str(number_of_frame_digits) + 'd.avi')


    subfolder = os.path.basename(file_in).replace('.avi', '-segmented')
    segmentation_dir = os.path.join(os.path.join(os.path.dirname(file_out), subfolder))

    if not os.path.exists(segmentation_dir):
        os.makedirs(segmentation_dir)

    file_out = os.path.join(segmentation_dir, os.path.basename(file_out))
    # file_out = os.path.join(os.path.dirname(file_out), os.path.basename(file_out))

    file_segment = file_out.replace('-chop%0'+ str(number_of_frame_digits) + 'd.avi', '-segments.csv')

    # Segment the input file by splitting the input file according to the frame numbers sequence
    # specified with the segment_frames option:
    # should be something like:
    # print('ffmpeg -i ../example_data/20171207_magnet.avi -codec copy -map 0 -f segment -segment_list ../example_data/20171207_magnet-segments.csv -segment_frames 100,200,300,500,800 ../example_data/20171207_magnet-chop%03d.avi')
    # cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy -map 0 -f '
    # cmd_string += 'segment -segment_list ' + file_segment + ' -segment_frames '+segmentation_frames+' ' + file_out

    cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy -map 0 -f '
    cmd_string += 'segment -segment_list ' + file_segment + ' -segment_frames ' + segmentation_frames + ' -reset_timestamps 1 ' + file_out



    print('start time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print(cmd_string)
    x = os.system(cmd_string)
    print('end time:\t{:s}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print('wrote:\n{:s}'.format(file_out))


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

def load_time_trace(source_folder_positions, video_file_name, methods=[], verbose=False):
    """
    Takes in the located in source_folder_positions
    from which containing the bead positions

    Args:
        source_folder_positions: folderpath positions data (.dat files obtained with track_sphere.extract_position_data)
        video_file_name: name of video file from which position data was extracted
        methods: list of methods used to extract position data

        verbose: if True print some output

    Returns: the position data as a pandas dataframe

    """


    assert video_file_name[-4:] == '.avi'

    filepath = os.path.join(source_folder_positions, video_file_name)

    if methods == []:
        print('define method!')

    info = {}
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

if __name__ == '__main__':
    folder_in = '../example_data/'
    filename_in = '20171207_magnet.avi'

    folder_in = '../raw_data/'
    # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    filename_in = '20180529_Sample6_bead_1_direct_thermal_01c.avi'
    # filename_in = '20180523_Sample6_bead_1_direct_thermal_03_reencode.avi'
    # filename_in = '20180524_Sample6_bead_1_direct_thermal_pump_isolated_10b.avi'


    file_in = os.path.join(folder_in, filename_in)


    ffmpeg_segment_video(file_in, 100)


    # fix key
    # file_out = file_in.replace('.avi', '-fixed.avi')
    # cmd_string = 'ffmpeg -i ' + file_in + ' -codec copy '
    # cmd_string += ' -reset_timestamps 1 ' + file_out
    #
    # print(cmd_string)
    # x = os.system(cmd_string)

