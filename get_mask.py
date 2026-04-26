import os
import numpy as np
import cv2
from ultralytics import YOLO

def extract_instance_masks(model_path, image_path, output_dir):
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 YOLO-seg 模型
    model = YOLO(model_path)
    
    # 推理
    results = model.predict(image_path, imgsz=640, conf=0.3)  # 降低置信度阈值
    
    # 保存模型检测后的结果图
    for result in results:
        detected_img = result.plot()  # 生成检测结果图
        detected_output_path = os.path.join(output_dir, "detected_image.png")
        cv2.imwrite(detected_output_path, detected_img[..., ::-1])
        print(f"Saved detected image to {detected_output_path}")
        print(f"Detected {len(result.boxes)} instances")
        print(f"Classes: {[result.names[int(cls)] for cls in result.boxes.cls]}")
        print(f"Boxes: {result.boxes.xyxy}")  # 检测框坐标
        print(f"Masks: {len(result.masks.xy)}")  # 掩码数量
    
    # 获取原图尺寸
    for result in results:
        orig_img = result.orig_img
        h, w = orig_img.shape[:2]
        
        # 检查是否检测到目标
        if result.masks is None:
            print("No instances detected in the image.")
            return
        
        # 初始化总掩码图（用于验证）
        total_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 遍历每个目标实例
        for idx, (mask, box) in enumerate(zip(result.masks.xy, result.boxes)):
            class_id = int(box.cls[0])
            class_name = result.names[class_id] if result.names else f"class_{class_id}"
            print(f"Processing instance {idx}, class: {class_name}, points: {len(mask)}")
            
            # 创建独立的黑色背景掩码图
            mask_image = np.zeros((h, w), dtype=np.uint8)
            
            # 将掩码多边形填充为白色
            mask_points = np.int32([mask])
            cv2.fillPoly(mask_image, mask_points, 255)
            
            # 叠加到总掩码图（用于调试）
            cv2.fillPoly(total_mask, mask_points, 255)
            
            # 保存独立掩码图
            output_filename = f"mask_{idx}_class_{class_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, mask_image)
            print(f"Saved mask for instance {idx} (class: {class_name}) to {output_path}")
        
        # 保存总掩码图（用于验证所有实例）
        total_mask_path = os.path.join(output_dir, "total_mask.png")
        cv2.imwrite(total_mask_path, total_mask)
        print(f"Saved total mask to {total_mask_path}")

if __name__ == '__main__':
    # 设置路径和参数
    model_path = r'H:\zxg\yolov8\ultralytics-main\yolov11-seg_best.pt'  # YOLO-seg 模型路径
    image_path = r'H:\zxg\chenchuibin\distortion\camera-correction-master\datasets\JPEGImages\1.jpg'  # 输入图像路径
    output_dir = r'H:\zxg\yolov8\ultralytics-main\output'  # 输出掩码图文件夹
    
    # 执行掩码提取
    extract_instance_masks(model_path, image_path, output_dir)