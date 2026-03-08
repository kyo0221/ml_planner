import pyzed.sl as sl


class ZED_API_Utils:
    def __init__(self):
        self.zed = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.pose = sl.Pose()
        self.translation = sl.Translation()
        self.orientation = sl.Orientation()

        self.init_param()
        self.init_tracking_param()

    def __del__(self):
        self.zed.disable_positional_tracking()
        self.zed.close()

    def init_param(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 15
        init.coordinate_units = sl.UNIT.METER
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(-1)

    def init_tracking_param(self):
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_area_memory = True
        tracking_params.enable_pose_smoothing = True
        tracking_params.enable_imu_fusion = False

        err = self.zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(-1)
    
    def grab(self):
        return self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS

    def get_image(self):
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(960 // 4, 600 // 4))
        img = self.image.get_data()

        return img
    
    def get_odom(self):
        state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.CAMERA)

        translation = self.pose.get_translation(self.translation).get()
        orientation = self.pose.get_orientation(self.orientation).get()

        return {
            'state': state,
            'timestamp_ns': self.pose.timestamp.get_nanoseconds(),
            'position': {
                'x': float(translation[0]),
                'y': float(translation[1]),
                'z': float(translation[2]),
            },
            'orientation': {
                'x': float(orientation[0]),
                'y': float(orientation[1]),
                'z': float(orientation[2]),
                'w': float(orientation[3]),
            },
            'pose_confidence': self.pose.pose_confidence(),
        }
