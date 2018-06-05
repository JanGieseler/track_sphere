# the goal of this file is to check if open cv works properly, i.e. if we can read and write files


import cv2 as cv
import os
import numpy as np

import matplotlib.pyplot as plt




################################################################################
#### SIFT
################################################################################
# img = cv2.imread('home.jpg')

# # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# filename = 'home.jpg'
# filename = 'magnet.jpg'
# gray=cv2.imread(filename)
#
#
# print(os.path.exists(filename))
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
#
# img=cv2.drawKeypoints(gray,kp, img)
#
# cv2.imwrite('sift_keypoints.jpg',img)
#
# img=cv2.drawKeypoints(gray,kp,img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv2.imwrite('sift_keypoints.jpg',img)



################################################################################
#### SURF
################################################################################

# img = cv2.imread('magnet.jpg')
#
# print(np.shape(img))
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.bitwise_not(img))
# surf = cv2.xfeatures2d.SURF_create(100)
# kp, des = surf.detectAndCompute(img,None)
#
#
# surf.setHessianThreshold(1000)
#
# kp, des = surf.detectAndCompute(img,None)
# print(len(kp))
#
#
#
# mask = np.zeros(np.shape(img)[:2], np.uint8)
# for k in kp:
#     print(k.angle, k.size, k.pt, (int(k.pt[0]), int(k.pt[1])))
#     x = cv2.circle(img, (int(k.pt[0]), int(k.pt[1])), 5, color = 100)
#
#     # mask[int(k.pt[0])-3:int(k.pt[0])+3,int(k.pt[1])-3:int(k.pt[1])+3] = 255
#
# mask = cv2.bitwise_not(mask)
# img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
#
#
# # plt.imshow(cv2.bitwise_not(img2)),plt.show()
#
# img3 = img* mask[:,:,np.newaxis]
#
# thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
# im2, contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#
# print('number of contours', len(contours))
# for i, c in enumerate(contours):
#     if len(c)>10:
#         print(i, len(c))
#
#
# contour_magnet = max(contours, key=len)
# print(len(contour_magnet))
#
# cv2.drawContours(img, [contour_magnet], -1, (0,255,0), 1)
#
# M = cv2.moments(contour_magnet)
# cX = int(M["m10"] / M["m00"])
# cY = int(M["m01"] / M["m00"])
#
# ellipse = cv2.fitEllipse(contour_magnet)
#
# cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)
#
# cv2.ellipse(img, ellipse, (0, 0, 255), 1)
#
# print('eeeee', ellipse)

# print('ellipse', ellipse)
# rotated_rect = cv2.ellipse(thresh,ellipse,(0,255,0), 2)
# print('asa', np.shape(rotated_rect))
# plt.imshow(img),plt.show()




################################################################################
#### Contours
################################################################################


# # import the necessary packages
# import argparse
# import imutils
# import cv2
#
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to the input image")
# args = vars(ap.parse_args())
#
# # load the image, convert it to grayscale, blur it slightly,
# # and threshold it
# image = cv2.imread(args["image"])
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
#
# # find contours in the thresholded image
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# # loop over the contours
# for c in cnts:
#     # compute the center of the contour
#     M = cv2.moments(c)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#
#     # draw the contour and center of the shape on the image
#     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#     cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
#     cv2.putText(image, "center", (cX - 20, cY - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#     # show the image
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)
# python center_of_shape.py --image shapes_and_colors.png


################################################################################
#### Optical flow
################################################################################

#
# from track_sphere.extract_data_opencv import features_surf
#
# method_parameters = {}
# if not 'xfeatures' in method_parameters:
#     method_parameters['xfeatures'] = 100
# if not 'HessianThreshold' in method_parameters:
#     method_parameters['HessianThreshold'] = 1000
# if not 'num_features' in method_parameters:
#     method_parameters['num_features'] = 5
#
#
# folder_in = '../example_data/'
# # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
# filename_in = '20171207_magnet.avi'
# file_in = os.path.join(folder_in, filename_in)
# cap = cv2.VideoCapture(file_in)
#
# # params for ShiTomasi corner detection
# feature_params = dict(maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
#
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# print('asasda',(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# # Create some random colors
# color = np.random.randint(0,255,(100,3))
#
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#
#
# p0b, _ = features_surf(old_gray, method_parameters, return_image=False)
# p0b = np.reshape(p0b, [method_parameters['num_features'], 1, 4])[:, :, :2]
# p0b = p0b.astype(np.int)
#
# # p0b = list(np.array(p0b))
# print('A', p0)
#
# print('B', np.array(p0b))
#
#
# print(np.shape(p0), np.shape(p0b))
# print(type(p0), type(p0b))
#
# p0 = p0b
#
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# counter = 0
# while(1):
#     # if counter >10:
#     #     break
#     counter += 1
#     ret,frame = cap.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calculate optical flowx`
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#
#
#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]
#
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv2.add(frame,mask)
#
#     cv2.imshow('frame',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
#
# cv2.destroyAllWindows()
# cap.release()



################################################################################
#### feature matching
################################################################################
# # give file to test

# frame1, frame2 = 100, 103
#
# folder_in = '../example_data/'
# filename_in = '20171207_magnet.avi'
#
# file_in = os.path.join(folder_in, filename_in)
#
#
#
# parameters = {}
# parameters['xfeatures'] = 20
# parameters['HessianThreshold'] = 1000
# parameters['num_features'] = 5
# parameters['winSize'] = (30, 30)
#
# cap = cv.VideoCapture(file_in)
#
#
# surf = cv.xfeatures2d.SURF_create(parameters['xfeatures'])
#
# surf.setHessianThreshold(parameters['HessianThreshold'])
#
# ds = []
# ks = []
# fs = []
#
#
#
#
# for f in [frame1,frame2]:
#     file_out = file_in.replace('.avi', 'test-{:d}.jpg'.format(f))
#     # read frame
#     cap.set(cv.CAP_PROP_POS_FRAMES, f)  # set the starting frame for reading to min frame
#     ret, frame = cap.read()
#
#     kp, des = surf.detectAndCompute(frame, None)
#     print('asdas', len(kp))
#
#     # Perform a simple blob detect
#     params = cv.SimpleBlobDetector_Params()
#     params.filterByArea = True
#     params.minArea = 20  # The dot in 20pt font has area of about 30
#     params.filterByCircularity = True
#     params.minCircularity = 0.1
#     params.filterByConvexity = True
#     params.minConvexity = 0.1
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.1
#     params.blobColor = 255
#
#
#
#
#     # Set up the detector with default parameters.
#     detector = cv.SimpleBlobDetector_create()
#
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Detect blobs.
#     keypoints = detector.detect(gray)
#
#     print('keypiits', keypoints)
#
#     # Draw detected blobs as red circles.
#     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#     im_with_keypoints = cv.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
#                                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#
#
#     # for k in kp:
#     #     # print(k.angle, k.size, k.pt, (int(k.pt[0]), int(k.pt[1])))
#     #     cv.circle(frame, (int(k.pt[0]), int(k.pt[1])), 5, color=100)
#     # frame = cv.drawKeypoints(frame, kp, None)
#
#     cv.imwrite(file_out, frame)
#
#     ds.append(des)
#     ks.append(kp)
#     fs.append(frame)
#
#
# # input('anykey...')
# cap.release()
# cv.destroyAllWindows()


################################################################################
#### contours in roi
################################################################################
from track_sphere.extract_data_opencv import lrc_from_features


folder_in = '../example_data/'
filename_in = '20171207_magnet.avi'

file_in = os.path.join(folder_in, filename_in)

features = [91.65412902832031, 102.88677978515625, 20.0, 144.75820922851562, 134.09725952148438, 105.93905639648438, 19.0, 122.29878234863281, 88.18359375, 55.885433197021484, 18.0, 148.00938415527344, 94.151123046875, 136.26904296875, 17.0, 211.0707550048828, 62.0274543762207, 101.42308044433594, 14.0, 122.03585052490234]

# radius = 30

h, w = 30, 26

lrcs = lrc_from_features(features, winSize=(h, w))


parameters = {}
parameters['threshold'] = 127
parameters['maxval'] = 255
# parameters = {}
# parameters['xfeatures'] = 20
# parameters['HessianThreshold'] = 1000
# parameters['num_features'] = 5
# parameters['winSize'] = (30, 30)

cap = cv.VideoCapture(file_in)

ds = []
ks = []
fs = []




for f in [1000,2]:
    file_out = file_in.replace('.avi', 'test-{:d}.jpg'.format(f))
    # read frame
    cap.set(cv.CAP_PROP_POS_FRAMES, f)  # set the starting frame for reading to min frame
    ret, image = cap.read()

    # frame = cv.seamlessClone(image, None)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    print('A', np.sum(image))

    lrc = lrcs[2]

    subimg = frame[lrc[1]:lrc[1] + h:, lrc[0]:lrc[0] + w]
    subimg_col = image[lrc[1]:lrc[1] + h:, lrc[0]:lrc[0] + w]

    # whole image
    # subimg = frame
    # subimg_col = image


    # subimg = frame[lrc[0]:lrc[0] + h+10:, lrc[1]:lrc[1] + w]
    print(np.shape(subimg))


    thresh = cv.threshold(subimg, parameters['threshold'], parameters['maxval'], cv.THRESH_BINARY)[1]

    print('aaa',np.shape(thresh))

    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(subimg, contours, -1, (0, 255, 0), 1)
    cv.drawContours(subimg_col, contours, -1, (255, 255, 0), 5)
    fs.append(frame)

    print('gggg', np.shape(image))
    print('B', np.sum(image))
    cv.imwrite(file_out, image)

    # print(contours)

    # print(im2)
    cv.imwrite(file_out, im2)


# input('anykey...')
cap.release()

cv.destroyAllWindows()