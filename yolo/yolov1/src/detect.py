# 检测文件，使用YOLOv1模型进行图片或者视频的检测，显示结果
# 包含 图片检测和视频检测两种模式
# 包含算法有 如何使用训练好的模型进行检测，NMS，opencv等等


import os
import cv2
import torch
import numpy as np
from model import YoloGoogleNet

# 第一步，将我们想要的输入变成模型可以接受的格式 现在我写的这个是448x448的图片

def preprocess_image(img):
    """
    预处理输入图像
    输入: BGR格式的图像
    输出: 归一化并调整大小后的张量
    """
    img_resized = cv2.resize(img, (448, 448))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    img_tensor = torch.tensor(img_transposed, dtype=torch.float32)
    # 这里可以加一个batch维度，但是我没加，如果你想做图片处理的加速，可以加上
    # img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 448, 448)
    return img_tensor

# 第二步：加载视频流或者图片


class ReadImg:
    def __init__(self, path):
        self.path = path
        self.mode = "image"
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        _, ext = os.path.splitext(path)
        if ext.lower() in video_extensions:
            self.mode = "video"

        if self.mode == "image":
            '''加载图片，先判断是文件夹还是单张的图片'''
            if os.path.isdir(path):
                self.img_files = [os.path.join(path, f) for f in os.listdir(path)
                                 if f.endswith('.jpg') or f.endswith('.png')]
                self.length = len(self.img_files)
            else:
                self.img_files = [path]
                self.length = 1
        else:
            '''加载视频'''
            self.cap = cv2.VideoCapture(path)
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        
    def __iter__(self):
        # 在迭代开始时重置索引（可选）
        if self.mode == 'image':
            self.index = 0
        return self

    def __next__(self):
        if self.mode == 'image':
            if self.index >= self.length:
                raise StopIteration
            img_path = self.img_files[self.index]
            self.index += 1
            img = cv2.imread(img_path)
            if img is None:
                # 跳过无法读取的文件或抛出
                return self.__next__()
            return preprocess_image(img)
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration
            return preprocess_image(frame)

# 第三步：加载模型进行推理

class Dectector:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = YoloGoogleNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, input_tensor):
        """
        输入: 预处理后的图像张量 (3, 448, 448)
        输出: 模型的预测结果
        """
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            input_tensor = input_tensor.unsqueeze(0)  # 添加batch维度
            output = self.model(input_tensor)
        return output.squeeze(0).cpu()  # 去掉batch维度并返回到CPU 这里的output是(7,7,11)

    


if __name__ == "__main__":
    # 简单测试
    detector = Dectector(model_path='path_to_your_model.pth', device='cpu')
    reader = ReadImg('')

    for img_tensor in reader:
        output = detector.predict(img_tensor)
        print(output.shape)  # 输出的形状应该是 (7, 7, 30) 对应YOLOv1的输出