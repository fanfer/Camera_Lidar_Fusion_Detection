import argparse
import os
from pathlib import Path
import sys
from fusion_function import *
from DataFrame import DataFrame
import numpy as np
from detect import *
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] #the root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
sys.path.insert(0,'../Fusion3Detect')
imgBeginIndex = 0

def run(
    weights = ROOT / 'models/yolov5s.pt',
    images_filename = ROOT / 'data/images', 
    lidars_filename = ROOT / 'data/lidars',
    imgsz = (640, 640),  # inference size (height, width)
    batch = 3,
    camera_lidar_weights = ROOT / 'src/calibration.txt',
    conf_thres = 0.7,
    shrink_rate = 0.1,
    alpha = 0.1,
    class_file = ROOT / 'models/class.txt',
    project=ROOT / 'runs/save',  # save results to project/name
    name='exp',  # save results to project/name
    nosave=False,
    visualize = True
        ):
    #calibration file for camera and lidar
    camera_matrix,camera_lidar_matrix =  read_calibration(camera_lidar_weights)
    camera_camera_matrix = np.diag([1,1,1,1])
    #images data
    images,image_names = read_images(images_filename,imgsz,imgBeginIndex,batch)
    #load 3D lidar points from file
    lidar_points_set = load_lidar_points_cloud(lidars_filename,imgBeginIndex,batch)
    
    yolov5 = YOLOV5(conf_thres=conf_thres,weights=weights)
    databuffer = []
    for index in range(batch):
        #load image into data frame buffer
        frame = DataFrame(images[index],image_names[index])
        frame.lidar_points = lidar_points_set[index]
        databuffer.append(frame)
        print("#1 load images into buffer done")
        
        #detect and classify objects
        yolov5.infer(frame.camera_img,frame=frame)
        print("#2 detect and classify objects done")
        
        #crop lidar points
        #remove lidar points based on distance properties
        corp_lidar_points(frame.lidar_points,minZ = -1.5, maxZ = 0.9, minX = 0.1, maxX = 20.0, maxY = 2.0, minR = 0.1)
        print("#3 crop lidar points done")
        
        #project lidar points to image
        #project_lidar_2_image(frame,camera_matrix,camera_lidar_matrix,camera_camera_matrix)
        
        #cluser the points based on bounding boxes
        cluster_lidar_with_ROI(shrink_rate,frame.bounding_boxes,frame,frame.camera_img,alpha,camera_matrix,camera_lidar_matrix,camera_camera_matrix)
        print("#4 cluser the points based on bounding boxes done") 
        
        save_show_result(frame,class_file, visualize, nosave,project, name,frame.img_name,camera_matrix,camera_lidar_matrix,camera_camera_matrix)
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'models/yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--images-filename', type=str, default=ROOT / 'data/images', help='images path')
    parser.add_argument('--lidars-filename', type=str, default=ROOT / 'data/lidars', help='lidar point cloud path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(640,640), help='inference size h,w')
    parser.add_argument('--batch',type=int, default=3, help='number of images to handle')
    parser.add_argument('--camera-lidar-weights', type=str, default=ROOT / 'src/calibration.txt', help='camera lidar calibration weights')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--shrink-rate', type=float, default=0.1, help='bounding box shrink rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha the weight of camera confidence')
    parser.add_argument('--class-file', type=str, default=ROOT / 'models/class.txt', help='classes path for the images')
    parser.add_argument('--project', default=ROOT / 'runs/save', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--visualize', action='store_false', help='visualize features')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    run(**vars(opt))
    
if __name__== '__main__':
    opt = parse_opt()
    main(opt)