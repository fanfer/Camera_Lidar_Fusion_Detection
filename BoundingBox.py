class BoundingBox:
    def __init__(self, boxID, classID,confidence=0, confidenceLidar=0 ,confidenceCamera=0):
        self.boxID = boxID #unique id for the bounding box
        self.roi = [] #2d region-of-interest in image coordinates [left,top , right,bottom]
        self.classID = classID # ID based on class file provided 
        self.confidence = confidence #the total confidence
        self.confidence_lidar = confidenceLidar #the confidence of lidar
        self.confidence_camera = confidenceCamera #the confidence of camera
        self.lidar_points = []
        