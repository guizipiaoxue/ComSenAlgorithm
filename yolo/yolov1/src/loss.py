# 实现损失函数
import torch
from torch import nn
from torch.nn import functional as F

# 注意这里在编写程序的时候，张量不再是data.py中的 S x S x (B*5 + C) 的形式
# 而是 batch x S x S x (B*5 + C) 的形式
# 具体看DataLoader类的说明

class YoloLoss(nn.Module):
    """
    YOLOv1 损失函数
    """
    def __init__(self):
        self.l_coord = 5.0  # 坐标损失权重
        self.l_noobj = 0.5  # 无目标损失权重

    def forword(self, p, a):
        # 计算预测边框与真实的边框的IOU
        iou = get_iou(p, a)  # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim = -1)[0]




    def get_iou(self,p, a):
        """
        计算预测的边框和真实的边框的IOU
        """

        # 首先获得四个角的坐标
        p_tl, p_br = self.get_box_corners(p)  # (batch, S, S, B, 2)
        a_tl, a_br = self.get_box_corners(a)  # (batch, S, S, B, 2)


        expanded_size = (-1, -1, -1, 2, 2, 2) # batch, S, S, B, B, 2
        tl  = torch.max(
            p_tl.unsqueeze(4).expand(expanded_size),  # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
            a_tl.unsqueeze(3).expand(expanded_size)  # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
        )
        br = torch.min(
            p_br.unsqueeze(4).expand(expanded_size), # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
            a_br.unsqueeze(3).expand(expanded_size) # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
        )

        # 计算交集和并集
        intersection_sides = torch.clamp(br - tl, min = 0.0) # 这一步









    def get_box_corners(self, b):
        """获取边框的四个角的坐标"""

        x_center = b[..., 0::5] # 这里不只是有一个边框，而是B个边框，所以要每隔5个取一个
        y_center = b[..., 1::5]
        width = b[..., 2::5]
        height = b[..., 3::5]

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2


        # 注意这里的张量是 （Batch, S, S, B, 2）
        return torch.stack((x1, y1), dim = 4), torch.stack((x2, y2), dim = 4) # 一步粘贴，具体可以直接右键转到torch定义




