from ultralytics import YOLO

# 加载预训练模型
# 添加注意力机制，SEAtt_yolov8.yaml 默认使用的是n。
# SEAtt_yolov8s.yaml，则使用的是s，模型。


# Use the model
if __name__ == '__main__':
    # Use the model
    model = YOLO(r"H:\zxg\yolov8\ultralytics-main\ultralytics\cfg\models\11\yolo11-seg.yaml").load(r'H:\zxg\yolov8\ultralytics-main\yolo11n-seg.pt')
    results = model.train(data=r'H:\zxg\yolov8\ultralytics-main\ultralytics\cfg\datasets\coco8-seg.yaml', epochs=1,
                          batch=1, workers=0)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')
    # batch的参数设置在results的括号里面