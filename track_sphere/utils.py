import os, yaml

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
    info = {key: data[0][key] for key in ['FrameRate', 'FileSize', 'Duration']}
    info.update({key: data[1][key] for key in ['Width', 'Height', 'BitDepth', 'CodecID', 'FrameCount']})

    # now convert to numbers
    info.update({key: int(info[key]) for key in ['FrameCount', 'BitDepth', 'FileSize', 'Width', 'Height']})
    info.update({key: float(info[key]) for key in ['FrameRate', 'Duration']})

    return info