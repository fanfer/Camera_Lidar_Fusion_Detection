class DataFrame:
    def __init__(self,img,img_name):
        self.camera_img = img #camera image
        self.img_name = img_name
        self.lidar_points = [] 
        self.bounding_boxes = []
        self.image_points = []