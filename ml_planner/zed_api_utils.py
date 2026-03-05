import pyzed.sl as sl


class ZEDApi:
    def __init__(self):
        self.zed = sl.Camera() 
        self.init_param()
        
        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()

    def __del__(self):
        self.zed.close()

    def init_param(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.SVGA
        init.camera_fps = 15

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(-1)
    
    def grab(self):
        return self.zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS

    def get_image(self):
        img = None
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.Resolution(960 // 4, 600 // 4))
        img = self.image.get_data()

        return img