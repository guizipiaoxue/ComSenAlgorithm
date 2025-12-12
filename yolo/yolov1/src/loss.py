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
        # p: 预测张量，形状为 (batch, S, S, B*5 + C)
        # a: 真实张量，形状为 (batch, S, S, B*5 + C)
        iou = self.get_iou(p, a)  # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim = -1)[0] # 这里我们取最大的IOU值 (batch, S, S, B)

        # 这里因为yolov1的计算公式中要对掩码进行处理
        # 处理confidence的掩码
        bbox_mask = p[..., 4::5] > 0.0 # (batch, S, S, B)  如果grid cell中有物体则为1，否则为0
        p_template = p[..., 4::5] > 0.0
        obj_i = bbox_mask[..., 0:1]  # 如果grid cell中有物体则为1，否则为0  (batch, S, S, 1)
        responsible = torch.zeros_like(p_template).scatter_(
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),
            value=1
        )  # (batch, S, S, B) 负责预测该物体的边框
        obj_ij = obj_i * responsible  # (batch, S, S, B) 负责预测且有物体的边框



        # X-Y位置损失
        x_losses = F.mse_loss(
            obj_ij * p[..., :10][..., 0::5],
            obj_ij * a[..., :10][..., 0::5]
        )

        y_losses = F.mse_loss(
            obj_ij * p[..., 1::5],
            obj_ij * a[..., 1::5]
        )

        pos_losses = x_losses + y_losses


        # 框的损失
        p_weight = p[..., 2::5]
        a_weight = a[..., 2::5]
        width_losses = F.mse_loss(
            # 这里使用sign函数来保持符号不变，p是预测的值，可能是负值
            # 不同的框对损失函数的贡献是不一样的，使用sqrt来减小大框的影响
            obj_ij * torch.sign(p_weight) * torch.sqrt(torch.abs(p_weight) + 1e-6),
            obj_ij * torch.sqrt(a_weight)
        )

        p_height = p[..., 3::5]
        a_height = a[..., 3::5]
        height_losses = F.mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + 1e-6),
            obj_ij * torch.sqrt(a_height)
        )

        boxx_losses = width_losses + height_losses

        # 置信度损失
        confidence_losses = F.mse_loss(
            obj_ij * p[..., 4::5],
            obj_ij * torch.ones_like(max_iou)
        )

        # 无目标置信度损失
        noobj_ij = ~obj_ij
        noobj_confidence_losses = F.mse_loss(
            noobj_ij * p[..., 4::5],
            torch.zeros_like(max_iou)
        )


        # 类别损失
        class_losses = F.mse_loss(
            # 这里我只有一个类别，如果报错自己修改
            obj_i * p[..., 10: ],
            obj_i * a[..., 10: ]
        )


        total = self.l_coord * (pos_losses + boxx_losses) \
                + confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses
        

        return total / p.size(0)  # 除以batch size
        

    def get_iou(self, p, a):
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
        intersection_sides = torch.clamp(br - tl, min = 0.0) # 这一步用来防止负值
        intersection = intersection_sides[..., 0] * intersection_sides[..., 1] # (batch, S, S, B, B)

        p_area = self.get_box_area(p)  # (batch, S, S, B)
        a_area = self.get_box_area(a)  # (batch, S, S, B)

        # 因为张量一步计算成IOU，所以需要扩展维度
        p_area = p_area.unsqueeze(4).expand_as(intersection)  # (batch, S, S, B, 1) -> (batch, S, S, B, B)
        a_area = a_area.unsqueeze(3).expand_as(intersection)  # (batch, S, S, 1, B) -> (batch, S, S, B, B)


        union = p_area + a_area - intersection

        # 防止IOU为分母为0
        replcement = torch.zeros_like(union) + 1e-6
        union = torch.where(union == 0.0, replcement, union)

        return intersection / union


    def get_box_corners(self, b):
        """获取边框的四个角的坐标"""
        # b 的形状是 (batch, S, S, B*5 + C)
        # 这里的 10 是因为 B=2, 2*5=10。如果 B 不同，这里需要修改
        # 为了安全，我们先去掉最后的类别部分
        box_data = b[..., :10] # (batch, S, S, 10)
        
        x_center = box_data[..., 0::5] # 0, 5
        y_center = box_data[..., 1::5] # 1, 6
        width = box_data[..., 2::5]    # 2, 7
        height = box_data[..., 3::5]   # 3, 8

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2


        # 注意这里的张量是 （Batch, S, S, B, 2）
        return torch.stack((x1, y1), dim = 4), torch.stack((x2, y2), dim = 4) # 一步粘贴，具体可以直接右键转到torch定义
    
    def get_box_area(self, b):
        """计算框的面积"""
        # 同样需要先截取，防止取到类别位
        box_data = b[..., :10]
        
        width = box_data[..., 2::5]
        height = box_data[..., 3::5]
        return width * height  # (batch, S, S, B)




if __name__ == "__main__":
    # 简单的单元测试
    torch.manual_seed(123)
    BATCH = 1
    S = 2
    B = 2
    C = 1 # 假设只有1个类别
    p = torch.zeros(BATCH, S, S, B*5 + C)
    a = torch.zeros(BATCH, S, S, B*5 + C)

    # 模拟一个真实标签
    # 在 (0,0) 网格有一个物体
    # box1 的真实值 (x,y,w,h,c)
    # 注意：通常在数据加载时，会将同一个GT复制给两个bbox的对应位置，或者只给其中一个
    # 这里我们假设两个slot都有GT数据，但只有confidence为1的才会被视为有物体
    a[0, 0, 0, 0:5] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
    a[0, 0, 0, 5:10] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0]) # 两个都给，看谁负责
    a[0, 0, 0, 10] = 1.0 # 类别

    # 预测值
    # box1 预测得比较准
    p[0, 0, 0, 0:5] = torch.tensor([0.45, 0.55, 0.25, 0.15, 0.8])
    # box2 预测得比较差
    p[0, 0, 0, 5:10] = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
    p[0, 0, 0, 10] = 0.8

    loss_fn = YoloLoss()
    
    # 注意：loss.py 中使用的是 forword 而不是 forward
    loss = loss_fn.forword(p, a)
    
    print(f"Input shape: {p.shape}")
    print(f"Total Loss: {loss.item()}")
    
    # 打印一些中间结果用于调试
    print("-" * 20)
    print("Debug Info:")
    iou = loss_fn.get_iou(p, a)
    print(f"IOU shape: {iou.shape}")
    print(f"Max IOU: {torch.max(iou, dim=-1)[0]}")
