import torch
from torch import nn
import torch.nn.functional as F


class YoloGoogleNet(nn.Module):
    """
    YOLOv1的骨干神经网络
    注意这里论文使用不是Resent，而是一个类似GoogleNet作为骨干网络
    
    """
    def __init__(self):
        super().__init__()
        # 我这里只有一个类，你要是要是有多个类就自己修改相关的代码
        self.depth = 2 * 5 + 1

        self.layers = [
            # 
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]


        for i in range(4):
            self.layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        self.layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(2):
            self.layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]

        self.layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1)
        ]


        # 全连接层
        self.layers += [
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, 7 * 7 * self.depth)
        ]

        self.model = nn.Sequential(*self.layers)


    def forward(self, x):
        return torch.reshape(self.model(x), (x.size(dim = 0), 7, 7, self.depth))    
    




if __name__ == "__main__":
    # 测试模型前向传播
    model = YoloGoogleNet()
    
    # 创建一个随机输入张量，模拟一张 448x448 的 RGB 图像
    # YOLOv1 通常使用 448x448 输入
    batch_size = 16
    input_tensor = torch.randn(batch_size, 3, 448, 448)
    
    # 运行前向传播
    output = model(input_tensor)
    
    # 打印输入和输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 7, 7, {model.depth})")
    
    # 验证输出形状是否正确
    expected_shape = (batch_size, 7, 7, model.depth)
    if output.shape == expected_shape:
        print("✓ Output shape is correct!")
    else:
        print("✗ Output shape mismatch!")
    
    # 打印一些输出统计信息
    print(f"Output min: {output.min().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    print(f"Output mean: {output.mean().item():.4f}")
