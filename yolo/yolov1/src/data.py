# image，label的录入，转换为相对张量

import os 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class YoloDataset(Dataset):
    """
    按照YOLOv1格式读取数据集
    """

    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        # 读取img和label文件名字
        self.img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.label_filenames = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        self.classes = {'red':0}  # 根据自己的数据集进行修改

    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        image, img = self.load_image(img_path)

        # 读取标签
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        label = self.load_label(label_path, img.size)

        return image, label, img
    
    def load_image(self, path):
        # 使用PIL加载图像，并转换为张量

        image = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])

        return transform(image), image

    def load_label(self, path, img_size):
        # 先读取标签文件
        w = img_size[0]
        h = img_size[1]
        bboxs = []
        classes = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                for i in range(len(parts)//5):
                    cls = int(parts[i*5])
                    x_center = float(parts[i*5 + 1])
                    y_center = float(parts[i*5 + 2])
                    width = float(parts[i*5 + 3])
                    height = float(parts[i*5 + 4])
                    classes.append(cls)
                    bboxs.append([x_center * w, y_center * h, width * w, height * h])

        s = 7  # 网格划分数
        b = 2  # 每个网格预测的边界框数
        c = 1  # 类别数  这里根据自己的数据集进行修改

        ground_truth = torch.zeros((s, s, b * 5 + c))


        # 先计算中心点所在的网格
        for i in range(len(bboxs)):
            x_center, y_center, width, height = bboxs[i]
            # 计算中心点所在的网格
            col = int(bboxs[i][0] * s // w)
            row = int(bboxs[i][1] * s // h)
            one_hot = torch.zeros(c)
            one_hot[classes[i]] = 1.0

            ground_truth[row, col, b*5:] = one_hot
            
            if col >= 0 and col <= s and row >= 0 and row <= s:
                # 计算相对于网格左上角的偏移量
                x_offset = (bboxs[i][0] * s / w) - col 
                y_offset = (bboxs[i][1] * s / h) - row
                # 计算相对于图像宽高的比例
                width_ratio = bboxs[i][2] / w
                height_ratio = bboxs[i][3] / h

                if i <= b - 1:
                    # 将标签信息存入网格
                    boxs = (
                        x_offset,
                        y_offset,
                        width_ratio,
                        height_ratio,
                        1.0  # 置信度
                    )
                    bbox_start = i * 5
                    ground_truth[row, col, bbox_start:bbox_start+5] = torch.tensor(boxs)

        return ground_truth


# 测试代码，路径以及数据使用自己的数据集和路径
if __name__ == "__main__":
    dataset = YoloDataset(img_dir='/home/py/Comsen算法组/yolo/yolov1/dataset/images', 
                          label_dir='/home/py/Comsen算法组/yolo/yolov1/dataset/labels')
    
    image, label, img = dataset[0]
    print("Image tensor shape:", image.shape)
    print("Label tensor shape:", label.shape)
    print(label)
    print("Image size:", img.size)

    # 显示图片
    # 这里也可以用PILLOW显示，然后tensor得先转换成numpy才能丢入cv2
    cv2.imshow('Image', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
