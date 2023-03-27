import random
import numpy as np
import os
from BoundingBox import BoundingBox
from LidarPoint import LidarPoint
import cv2
#读取标定结果
def read_calibration(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Locate the calibration matrix lines by searching for the keyword 'camera'
    start_ix = lines.index('camera\n') + 1
    end_ix = start_ix + 3

    # Read the 4x4 matrix located at the specified lines
    camera_matrix = np.zeros((3, 4))
    for i in range(start_ix, end_ix):
        camera_matrix[i-start_ix] = [float(x) for x in lines[i].split()]
        
    # Locate the calibration matrix lines by searching for the keyword 'calibration'
    start_ix = lines.index('calibration\n') + 1
    end_ix = start_ix + 4
    camera_lidar_matrix = np.zeros((4,4))
    for i in range(start_ix, end_ix):
        camera_lidar_matrix[i-start_ix] = [float(x) for x in lines[i].split()]

    return camera_matrix,camera_lidar_matrix


#读取图片数据
def read_images(folder_path,imgsz,imgBeginIndex,batch):
    image_names = os.listdir(folder_path)[imgBeginIndex:imgBeginIndex+batch] # 获取前batch张图片的文件名
    image_names.sort()    
    images = []
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)

        # 使用cv2读取图片
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # 使用cv2修改图片尺寸
        img = cv2.resize(img,imgsz)

        images.append(img)

    return images,image_names


#目标检测
def detect_objects(model_path,class_path,confidence_threshold,frame):
    
    image = frame.image
    # 加载YOLOv5模型和类别文件
    model = cv2.dnn.readNetFromTorch(model_path)
    classes = []
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, swapRB=True, crop=False)
    
    # 将blob送入模型进行检测
    model.setInput(blob)
    detections = model.forward()
    
    # 遍历检测到的目标并进行标注
    height, width, _ = image.shape
    index = 0
    for detection in detections:
        confidence = detection[4]
        if confidence > confidence_threshold:
            class_id = int(detection[5])
            class_name = classes[class_id]
            left = int(detection[0] * width)
            top = int(detection[1] * height)
            right = int(detection[2] * width)
            bottom = int(detection[3] * height)
            bounding_box = BoundingBox(index,class_id,confidenceCamera = confidence)
            bounding_box.roi = [left,top,right,bottom]
          
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), thickness=2)
            cv2.putText(image, class_name, (left + 5, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame.bounding_boxes.append(bounding_box)
            frame.camera_img = image
    return


#读取雷达点云数据
def load_lidar_points_cloud(lidars_filename,imgBeginIndex,batch):
    lidar_points_set = []
    lidar_names = os.listdir(lidars_filename)[imgBeginIndex:imgBeginIndex+batch]
    for lidar_name in lidar_names:
        lidar_points = []
        file_name = os.path.join(lidars_filename,lidar_name)
        pointcloud = np.fromfile(file_name, dtype=np.float32, count=-1).reshape([-1, 4])
        for point in pointcloud:
            lidar_point = LidarPoint(point[0],point[1],point[2],point[3])
            lidar_points.append(lidar_point)
        lidar_points_set.append(lidar_points)
    return lidar_points_set


#得到感兴趣区域的点云
def corp_lidar_points(lidar_points,minX,maxX,maxY,minZ,maxZ,minR):
    new_lidar_points = []
    for point in lidar_points :
        if point.x>=minX and point.x<=maxX and point.z>=minZ and point.z<=maxZ and abs(point.y)<=maxY and point.r>=minR :
            new_lidar_points.append(point)
    lidar_points = new_lidar_points
    return


#将点云投影到平面上
def project_lidar_2_image(point,camera_matrix,camera_lidar_matrix,camera_camera_matrix):
    X = np.zeros((4,1))
    #image_points = frame.image_points
    #lidar_points = frame.lidar_points
    
    #for point in lidar_points:
    X[0,0] = point.x
    X[1,0] = point.y
    X[2,0] = point.z
    X[3,0] = 1
        
    Y =  camera_matrix @ camera_camera_matrix @ camera_lidar_matrix @ X
    point = [Y[0,0]/Y[2,0],Y[1,0]/Y[2,0]]

    return point

  
def points_in_rect(lidar_points, top_left, bottom_right,camera_matrix,camera_lidar_matrix,camera_camera_matrix):
    # 转换矩形框的坐标为 NumPy 数组
    rect_corners = np.array([top_left, (bottom_right[0], top_left[1]),
                             bottom_right, (top_left[0], bottom_right[1])])
    # 存储在矩形框内的点
    points_in_rect = []
    
    rect_corners = rect_corners.astype(np.float32)

    # 遍历点
    for lidar_point in lidar_points:
        point = project_lidar_2_image(lidar_point,camera_matrix,camera_lidar_matrix,camera_camera_matrix)
        if np.isnan(point[0]):
            continue
        # 使用 cv2.pointPolygonTest 函数判断点是否在矩形框内
        point =tuple(int(a) for a in point )
        result = cv2.pointPolygonTest(rect_corners, point, False)
        
        # 判断点是否在矩形框内
        if result >= 0:
            points_in_rect.append(lidar_point)
    
    return points_in_rect


#calculate the center of the object
def calculate_center(lidar_points):
    point_x,point_y,point_z = 0,0,0
    for point in lidar_points:
        point_x = point.x + point_x
        point_y = point.y + point_y
        point_z = point.z + point_z
    
    point_x = point_x/len(lidar_points)
    point_y = point_y/len(lidar_points)
    point_z = point_z/len(lidar_points)
    
    return (point_x,point_y,point_z)


#根据bounding box 完成对点云的聚类
def cluster_lidar_with_ROI(shrink_rate,bounding_boxes,frame,image,alpha,camera_matrix,camera_lidar_matrix,camera_camera_matrix):
    num_points = len(frame.lidar_points)
    total_area = image.shape[0] * image.shape[1]
    for bounding_box in bounding_boxes :
         left = bounding_box.roi[0]
         bottom = bounding_box.roi[1]
         right = bounding_box.roi[2]
         top = bounding_box.roi[3]
         
         width = right - left
         height = top - bottom
         
         left = left - (width*shrink_rate)
         right = right + (width*shrink_rate)
         top = top + (height*shrink_rate)
         bottom = bottom - (height*shrink_rate)
         top_left = (left , top)
         bottom_right = (right , bottom)
         
         bounding_box.lidar_points = points_in_rect(frame.lidar_points,top_left,bottom_right,camera_matrix,camera_lidar_matrix,camera_camera_matrix)
         #center = calculate_center(bounding_box.lidar_points)
         #print(f"the center is {center}")
         num_points_in_roi = len(bounding_box.lidar_points)
         rect_area = width * height
         #计算置信度
         bounding_box.confidence_lidar = (num_points_in_roi*total_area)/(num_points*rect_area)
         bounding_box.confidence = alpha*bounding_box.confidence_lidar + ((1-alpha)*bounding_box.confidence_camera)
    return    

#保存或者展示结果
def save_show_result(frame,class_path, visualize, nosave,project, name,images_filename,camera_matrix,camera_lidar_matrix,
                     camera_camera_matrix):
    classes = []
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
       
    image = frame.camera_img
    #TODO 在图片上画上激光雷达点和矩形框、目标检测的类别等信息
    for bounding_box in frame.bounding_boxes:
        x = bounding_box.roi
        c1,c2 = (x[0],x[1]),(x[2],x[3])
        color = None
        color = color or [random.randint(0, 255) for _ in range(3)]
        tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        class_id = int(bounding_box.classID)
        class_name = classes[class_id-1]
        label =f'{class_name}:{bounding_box.confidence:.2f}'
        line_thickness=3 
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1 
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        #cv2.putText(image, s, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [225, 255, 255], 2)
        for point in bounding_box.lidar_points:
            val = point.x
            maxVal = 20
            red = min(255, (int)(255 * abs((val - maxVal) / maxVal)))
            green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))))
            img_point = project_lidar_2_image(point,camera_matrix,camera_lidar_matrix,camera_camera_matrix)
            if np.isnan(img_point[0]):
                continue
            img_point =tuple(int(a) for a in img_point )
            cv2.circle(image,img_point,1,(0,green,red),-1)
    
    if visualize:
        cv2.imshow("Fusion Result",image)
        cv2.waitKey(0) 

    if nosave :
        return
    file_path = os.path.join(project,name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    cv2.imwrite(os.path.join(file_path,images_filename),image)