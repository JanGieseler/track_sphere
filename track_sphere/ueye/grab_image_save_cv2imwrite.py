from pyueye import ueye
from _ctypes import sizeof
#customer Code

import numpy as np
import cv2
import time

import os
from contextlib import contextmanager
import itertools as it


def check(ret):
    if ret != ueye.IS_SUCCESS: raise BaseException(ret)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def main():

    cam = ueye.HIDS()   # open first available cam
    check(ueye.is_InitCamera(cam, None))

    # query image size
    size = ueye.IS_SIZE_2D()
    check((ueye.is_AOI(cam, ueye.IS_AOI_IMAGE_GET_SIZE, size, ueye.sizeof(size))))
    width = size.s32Width
    height = size.s32Height

    # allocate memory. we need at least one buffer
    mem_ptr = ueye.c_mem_p()
    mem_id = ueye.INT()
    bits_per_pixel = ueye.INT(24)   # assume we have 24 bits per pixel (rgb)
    check(ueye.is_AllocImageMem(cam, width, height, bits_per_pixel, mem_ptr, mem_id))

    # set the image mem active, so we can render into
    check(ueye. is_SetImageMem(cam, mem_ptr, mem_id))

    # we need to get the mem pitch for later usage
    pitch = ueye.INT()
    check(ueye.is_InquireImageMem(cam, mem_ptr, mem_id, width, height,  bits_per_pixel,pitch))
    #expMax = ueye.DOUBLE(0.0)
    expMax = [10,20,30,40]
    # now lets capture a contineous video
    while True:

        # For loop
        for i in my_range(0, 4, 1):

            if i == 4:
                break
            # print("i= ",expMax[i])
            timeExp = ueye.DOUBLE(expMax[i])
            check(ueye.is_Exposure(cam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, timeExp, ueye.sizeof(timeExp)))
            print("Set Exposure :", timeExp);
            time.sleep(1)

            # lets capture one image
            check(ueye.is_FreezeVideo(cam, ueye.IS_WAIT))

            # for this we use a function from the pyueye interface get data
            array = ueye.get_data(mem_ptr, width, height, bits_per_pixel, pitch, copy=False) # we do not want to copy
            # print(array)
            # we have to reshape the array

            frame = np.reshape(array, (height.value, width.value, 3))
            # print(frame)

            framesmall = cv2.resize(frame,(0,0),fx=0.3, fy=0.3)
            time.sleep(1)
            #cv2.imshow('img', framesmall)

            # Save image OpenCV
            cv2.imwrite("img_" + str(expMax[i]) + 'ms.jpg', frame)
            print("save image to file: " + "img_" + str(expMax[i]) + 'ms.jpg')


        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF

        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break


    # we have to free our image mem at the end
    check(ueye.is_FreeImageMem(cam, mem_ptr, mem_id))

    check(ueye.is_ExitCamera(cam))



if __name__ == "__main__":
    main()