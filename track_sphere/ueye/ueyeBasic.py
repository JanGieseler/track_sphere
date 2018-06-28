from pyueye import ueye
hcam = ueye.HIDS(0)
pccmem = ueye.c_mem_p()
memID = ueye.c_int(0)
ueye.is_InitCamera(hcam, None)
sensorinfo = ueye.SENSORINFO()
ueye.is_GetSensorInfo(hcam, sensorinfo)
ueye.is_SetColorMode(hcam, ueye.IS_CM_SENSOR_RAW8)
ueye.is_AllocImageMem(hcam, sensorinfo.nMaxWidth, sensorinfo.nMaxHeight, 8, pccmem, memID)
ueye.is_SetImageMem(hcam, pccmem, memID)

ueye.is_SetExternalTrigger(hcam, ueye.IS_SET_TRIGGER_SOFTWARE)
wertSet = ueye.c_double(1)
nret=ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, wertSet, ueye.sizeof(wertSet))
print(wertSet)

help(ueye.is_Exposure)

wert=ueye.c_double()
sizeo=ueye.sizeof(wert)
nret=ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, wert, sizeo)
print(wert)
nret = ueye.is_FreezeVideo(hcam, ueye.IS_WAIT)
print(nret)
FileParams = ueye.IMAGE_FILE_PARAMS()
FileParams.pwchFileName = "c:\python-test-image.png"
FileParams.nFileType = ueye.IS_IMG_PNG
FileParams.ppcImageMem = None
FileParams.pnImageID = None
nret = ueye.is_ImageFile(hcam, ueye.IS_IMAGE_FILE_CMD_SAVE, FileParams, ueye.sizeof(FileParams))
print(nret)

nret = ueye.is_FreezeVideo(hcam, ueye.IS_WAIT)
print(nret)
FileParams = ueye.IMAGE_FILE_PARAMS()
FileParams.pwchFileName = "c:\python-test-image1.png"
FileParams.nFileType = ueye.IS_IMG_PNG
FileParams.ppcImageMem = None
FileParams.pnImageID = None
nret = ueye.is_ImageFile(hcam, ueye.IS_IMAGE_FILE_CMD_SAVE, FileParams, ueye.sizeof(FileParams))
print(nret)
ueye.is_FreeImageMem(hcam, pccmem, memID)
ueye.is_ExitCamera(hcam)