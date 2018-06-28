from pyueye import ueye
import numpy
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QPushButton, QVBoxLayout, QApplication, QMessageBox
import sys
import platform
if platform.system() == 'Windows':
    import win32event


class ImageData:
    memory_pointer = None
    memory_id = None
    width = None
    height = None
    bits_per_pixel = None


class CaptureVideoWidget(QWidget):
    def __init__(self, parent=None):
        self.converted_memory_pointer = ueye.c_mem_p()
        self.converted_memory_id = ueye.int()
        self.img_data = ImageData()
        self.capturing = False
        self.hCam = 0
        self.init()
        ueye.is_SetColorMode(self.hCam, ueye.IS_CM_RGB8_PACKED)
        self.alloc(self.hCam)
        self.qt_image = None
        ueye.is_CaptureVideo(self.hCam, Wait=True)
        QWidget.__init__(self, parent, flags=Qt.Widget)
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.graphics_scene)
        self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.start_stop_button = QPushButton("start/stop")
        self.event = ueye.HANDLE(int())
        self.frame_event_id = ueye.IS_SET_EVENT_FRAME
        self.image_data_copy = None
        self.pil_image = None
        self.pix_map = None
        self.width = 0
        self.height = 0
        self.threads = []
        layout = QVBoxLayout()
        layout.addWidget(self.start_stop_button, alignment=Qt.AlignTop)
        layout.addWidget(self.graphics_view)
        self.setLayout(layout)
        self.start_stop_button.clicked.connect(self.switch_capturing)

    def init(self):

        h_cam = ueye.HIDS(0 | ueye.IS_USE_DEVICE_ID)
        ret = ueye.is_InitCamera(h_cam, None)

        if ret != ueye.IS_SUCCESS:
            error_message = QMessageBox()
            result = QMessageBox.critical(error_message, 'Error', "Invalid device id!", buttons=QMessageBox.Ok)
            if result == QMessageBox.Ok:
                exit()

        self.hCam = h_cam
        return h_cam

    def alloc(self, h_cam):
        rect_aoi = ueye.IS_RECT()
        memory_id = ueye.int()
        memory_pointer = ueye.c_mem_p()
        bits_per_pixel = 8
        ueye.is_AOI(h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        ueye.is_AllocImageMem(h_cam, rect_aoi.s32Width, rect_aoi.s32Height, bits_per_pixel, memory_pointer, memory_id)
        ueye.is_SetImageMem(h_cam, memory_pointer, memory_id)
        self.img_data.memory_pointer = memory_pointer
        self.img_data.memory_id = memory_id
        self.img_data.width = rect_aoi.s32Width
        self.img_data.height = rect_aoi.s32Height
        self.img_data.bits_per_pixel = bits_per_pixel
        ueye.is_AllocImageMem(h_cam, rect_aoi.s32Width, rect_aoi.s32Height, 24,
                              self.converted_memory_pointer, self.converted_memory_id)

    def closeEvent(self, event):
        self.capturing = False
        ueye.is_DisableEvent(self.hCam, self.frame_event_id)
        ueye.is_ExitEvent(self.hCam, self.frame_event_id)
        if ueye.is_CaptureVideo(self.hCam, ueye.IS_GET_LIVE):
            ueye.is_StopLiveVideo(self.hCam, Wait=True)

    def switch_capturing(self):

        if self.capturing:
            ueye.is_StopLiveVideo(self.hCam, Wait=True)
            self.capturing = False
        else:
            ueye.is_CaptureVideo(self.hCam, Wait=True)
            self.capturing = True
            self.display_image()

    def convert_image_data(self):
        rect_aoi = ueye.IS_RECT()
        bits_per_pixel = 24
        converted_image_data = ImageData()
        conversion_params = ueye.BUFFER_CONVERSION_PARAMS()
        ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        converted_image_data.memory_pointer = self.converted_memory_pointer
        converted_image_data.memory_id = self.converted_memory_id
        converted_image_data.width = rect_aoi.s32Width
        converted_image_data.height = rect_aoi.s32Height
        converted_image_data.bits_per_pixel = bits_per_pixel
        conversion_params.nDestPixelFormat = ueye.IS_CM_RGB8_PACKED
        conversion_params.pSourceBuffer = self.img_data.memory_pointer
        conversion_params.pDestBuffer = converted_image_data.memory_pointer
        conversion_params.nDestPixelConverter = ueye.IS_CONV_MODE_SOFTWARE_3X3
        conversion_params.nDestColorCorrectionMode = ueye.IS_CCOR_DISABLE
        conversion_params.nDestGamma = ueye.INT(100)
        conversion_params.nDestSaturationU = ueye.INT(100)
        conversion_params.nDestSaturationV = ueye.INT(100)
        conversion_params.nDestEdgeEnhancement = ueye.INT(0)
        ueye.is_Convert(self.hCam, ueye.IS_CONVERT_CMD_APPLY_PARAMS_AND_CONVERT_BUFFER,
                        conversion_params, ueye.sizeof(conversion_params))

        return converted_image_data

    def free_image_mem(self, mempointer, memid):
        ueye.is_FreeImageMem(self.hCam, mempointer, memid)

    def display_image(self):
        fps = ueye.DOUBLE()
        ueye.is_GetFramesPerSecond(self.hCam, fps)
        timeout = int((5/fps)*1000)
        h_event = None
        if platform.system() == 'Windows':
            h_event = win32event.CreateEvent(None, False, False, None)
            self.event = ueye.HANDLE(int(h_event))
            ueye.is_InitEvent(self.hCam, self.event, self.frame_event_id)
        ueye.is_EnableEvent(self.hCam, self.frame_event_id)

        while True:
            ret = None
            if not self.capturing:
                break
            if platform.system() == 'Windows':
                ret = win32event.WaitForSingleObject(h_event, timeout)
            elif platform.system() == 'Linux':
                ret = ueye.is_WaitEvent(self.hCam, self.frame_event_id, timeout)

            if ret == 0:
                converted_image_data = self.convert_image_data()

                self.image_data_copy = (ueye.CHAR * int(self.img_data.width * self.img_data.height * 3))()
                ueye.is_CopyImageMem(hCam=self.hCam, pcSource=converted_image_data.memory_pointer,
                                     nID=converted_image_data.memory_id, pcDest=self.image_data_copy)
                bytes_per_pixel = 3
                self.image_data_copy = numpy.reshape(self.image_data_copy,
                                                     (int(self.img_data.height), int(self.img_data.width),
                                                      bytes_per_pixel))
                self.image_data_copy = self.image_data_copy.view(numpy.uint8)
                self.pil_image = Image.fromarray(self.image_data_copy)
                self.graphics_scene.clear()
                self.width, self.height = self.pil_image.size
                self.qt_image = ImageQt.ImageQt(self.pil_image)
                self.pix_map = QPixmap.fromImage(self.qt_image)
                self.graphics_scene.addPixmap(self.pix_map)
                self.graphics_view.fitInView(QRectF(0, 0, self.width, self.height), Qt.KeepAspectRatio)
                self.graphics_scene.update()
                app.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = CaptureVideoWidget()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec_())
