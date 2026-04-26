import cv2
import numpy as np
from ultralytics import YOLO

# 加载训练好的 YOLOv11-seg 模型
model = YOLO(r"H:\zxg\yolov8\ultralytics-main\yolov11-seg_best.pt")  # 替换为你的模型权重路径

# 加载输入图像
image_path = r"H:\zxg\chenchuibin\distortion\camera-correction-master\images\1.jpg"  # 替换为你的输入图像路径
image = cv2.imread(image_path)
if image is None:
    raise ValueError("无法加载图像，请检查图像路径！")

# 进行实例分割预测
results = model.predict(image, conf=0.25, iou=0.45)  # conf和iou可根据需要调整

# 创建一个全黑的背景图像，尺寸与输入图像相同
height, width = image.shape[:2]
black_background = np.zeros((height, width, 3), dtype=np.uint8)

# 获取分割掩码并处理
for result in results:
    if result.masks is not None:  # 确保有分割掩码
        masks = result.masks.data.cpu().numpy()  # 获取所有掩码
        for mask in masks:
            # 将掩码调整为与输入图像相同的尺寸
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            # 将掩码转换为二值化（0或1）
            mask = (mask > 0).astype(np.uint8)
            # 使用掩码提取目标区域
            masked_region = cv2.bitwise_and(image, image, mask=mask)
            # 将目标区域叠加到黑色背景上
            black_background = cv2.bitwise_or(black_background, masked_region)

# 保存输出图像
output_path = r"H:\zxg\yolov8\ultralytics-main\output2\output_image.jpg"
cv2.imwrite(output_path, black_background)
print(f"处理后的图像已保存到 {output_path}")

# 可选：显示结果
cv2.imshow("Segmented Image", black_background)
cv2.waitKey(0)
cv2.destroyAllWindows()