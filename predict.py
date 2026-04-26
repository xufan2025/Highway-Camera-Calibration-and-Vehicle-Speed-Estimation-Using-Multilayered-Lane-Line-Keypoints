from ultralytics import YOLO

if __name__=="__main__":
    
    pth_path=r"H:\zxg\yolov8\ultralytics-main-seg\yolov11-seg_best.pt"

    test_path=r""

    model = YOLO(pth_path)  # load a custom model

    results = model(test_path,save=True,conf=0.5)