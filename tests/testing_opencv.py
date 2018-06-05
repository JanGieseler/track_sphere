# the goal of this file is to check if open cv works properly, i.e. if we can read and write files


import cv2 as cv
import os
import numpy as np



folder_in = '../example_data/'
filename_in = '20171207_magnet.avi'

file_in = os.path.join(folder_in, filename_in)

frame_id = 0
file_out = file_in.replace('.avi', '-test-{:d}.jpg'.format(frame_id))

print('START TEST')
print('{:s} => {:s}'.format(file_in, file_out))
cap = cv.VideoCapture(file_in)
cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)  # set the starting frame for reading to min frame
# read frame
ret, image = cap.read()

print('read frame', ret, np.shape(image))
# convert to gray image
frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.imwrite(file_out, image)
print('wrote image to file: {:s}'.format(file_out))

# input('anykey...')
cap.release()

cv.destroyAllWindows()

print('FINISHED TEST')
