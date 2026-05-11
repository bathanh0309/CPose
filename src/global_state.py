# chia sẻ dữ liệu giữa 2 camera

class GlobalState:
    cam1_data = []
    cam2_data = []
    
    @classmethod
    def update_cam1(cls, data):
        cls.cam1_data = data
        
    @classmethod
    def update_cam2(cls, data):
        cls.cam2_data = data