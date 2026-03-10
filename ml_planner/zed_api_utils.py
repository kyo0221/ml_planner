import pyzed.sl as sl


class ZED_API_Utils:
    def __init__(self):
        self.zed = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()

        self.init_param()

    def __del__(self):
        self.zed.close()

    def init_param(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 15
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(-1)
    
    def grab(self):
        return self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS

    def get_image(self):
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.Resolution(960 // 4, 600 // 4))
        img = self.image.get_data()

        return img