# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture('vtest.avi')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print('ext')
#         break
#
# cap.release()
# cv2.destroyAllWindows()

import numpy as np
import cv2 as cv

from tqdm import tqdm

import yaml, os


from time import sleep

def load_video_info(filename):
    """
    loads the video metadata that has been exported with metainfo (www.MediaArea.net) into a json file

    Args:
        filename: path to the .avi or .xml file

    Returns: ['FileSize', 'FrameRate', 'BitDepth', 'Width', 'Duration', 'FrameCount', 'Height', 'CodecID']

    """
    if len(filename.split('.avi'))==2:
        filename = filename.replace('.avi', '.xml')

    assert os.path.exists(filename)
    print(filename)

    with open(filename, 'r') as infile:
        data = yaml.safe_load(infile)

    data = data['media']['track']
    info = {}
    # select the relevant paramters
    info = {key: data[0][key] for key in ['FrameRate', 'FrameCount', 'FileSize', 'Duration']}
    info.update({key: data[1][key] for key in ['Width', 'Height', 'BitDepth', 'CodecID']})

    # now convert to numbers
    info.update({key: int(info[key]) for key in ['FrameCount', 'BitDepth', 'FileSize', 'Width', 'Height']})
    info.update({key: float(info[key]) for key in ['FrameRate', 'Duration']})

    return info





def substract_background(file_in, file_out=None, min_frame = 0, max_frame = None, fourcc = None, output_images = 1000, buffer_time=1e-6):
    """
    Takes a video file and outputs a new file where the background is substracted
    Args:
        file_in:
        file_out:


        fourcc: four character code for video type (https://www.fourcc.org)
            if None use the same as the input video

            some options are
            // Lossless in quality with good processing performance
            'LAGS'
            // no compression. WARNING can take up a lot of space!!
            'FULL'

        output_images: (int) if specified output an image every output_images iterations

        buffer_time:  wait time between frames, needed because of some buffer issues of opencv

    Returns:

    """



    ################################################################################
    #### checks the validity of the inputs and checks for existing files
    ################################################################################

    assert isinstance(file_in, str)


    if file_out is None:
        file_out = file_in.replace('.avi', '-no_bkgng.avi')


    if os.path.exists(file_out):
        print('output file exists. Abort operation')
        return None

    img_dir = file_out.replace('.avi', '-img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    #get the metadata from the input file
    info = load_video_info(file_in)

    # if not specified, get code from input file
    if fourcc is None:
        fourcc = info['CodecID']

    assert isinstance(fourcc, str)
    assert len(fourcc)==4

    if max_frame is None:
        max_frame = info['FrameCount']

    ################################################################################
    #### setup input and output streams
    ################################################################################

    cap = cv.VideoCapture(file_in) #open input video
    fgbg = cv.createBackgroundSubtractorMOG2() # create background substractor

    if fourcc == 'FULL':
        # no compression
        fourcc = 0
    else:
        fourcc = cv.VideoWriter_fourcc(*fourcc)

    # last argument means that we load a black and white image
    video_writer = cv.VideoWriter(file_out, fourcc, info['FrameRate'], (info['Width'], info['Height']), False)


    ################################################################################
    #### start processing
    ################################################################################
    print('subtracting background: {:s} => {:s}'.format(file_in,file_out))
    print('frames {:d}-{:d} ({:d})'.format(min_frame, max_frame, max_frame-min_frame))

    for frame_idx in tqdm(range(max_frame)):

        ret, frame_in = cap.read()
        if ret:
            frame_out = fgbg.apply(frame_in)
            # cv.imshow('frame', frame_out)

            if buffer_time>0:
                sleep(buffer_time)
            video_writer.write(frame_out)

        if output_images>0 and frame_idx%output_images==0:
            # print('writing',frame_idx)

            cv.imwrite(os.path.join(img_dir,file_out.replace('.avi', '-{:d}.jpg'.format(frame_idx))), frame_out)



        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ################################################################################
    #### clean up
    ################################################################################
    video_writer.release()
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # substract_background('test.avi', file_out=None)
    info = load_video_info('test.avi')
    print(info)
