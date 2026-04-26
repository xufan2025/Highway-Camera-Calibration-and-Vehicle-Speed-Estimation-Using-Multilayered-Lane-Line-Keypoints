import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__=="__main__":
    model = YOLO(r'H:\zxg\yolov8\ultralytics-main\runs\detect\train18\weights\best.pt')
    model.val(data=r'H:\zxg\yolov8\ultralytics-main\datasets\mydata.yaml',
              imgsz = 640,
              batch = 8,
              split = 'val',
              workers = 0,
              device = '0',
              )