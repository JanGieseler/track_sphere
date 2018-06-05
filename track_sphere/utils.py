import os, yaml
import numpy as np
import cv2 as cv

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