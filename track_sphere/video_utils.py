import os

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


def extract_video_info(filepath, media_info_file_path='C:/Program Files/MediaInfo/CLI/', verbose=False):
    """
    uses the commanline interface from media_info to extract metadata from a video file and writes it to a json file
    with the same name as the video file (except for the file extension)
    Args:
        filepath: path to video file
        media_info_file_path: location of .exe file of mediainfo CLI
            (download here: https://mediaarea.net/en/MediaInfo/Download/Windows)

    Returns:

    """


    # video_folder = 'Z:/Lab/Lev/videos/20180607\_Sample\_6\_bead\_1'
    # video_file = '20180607_Sample_6_Bead_3.avi'

    # filepath = os.path.join(video_folder, video_file)

    media_info_file_path = media_info_file_path.replace(' ', '^ ') # in windows spaces have to be escaped with ^

    cmd_string = media_info_file_path + 'MediaInfo.exe --Output=JSON '
    cmd_string += filepath + ' > ' + filepath.replace('.avi', '.json')

    cmd_string = cmd_string.replace('\_', '_')  # for command line output we actually don't want the escape
    if verbose:
        print(cmd_string)
    x = os.system(cmd_string)



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

