# 训练模型

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm

from data import YoloDataset
from model import YoloGoogleNet
from loss import YoloLoss


class trainer:
    """编写训练器"""
    def __init__(self, device):
        self.set_args()
        self.device = device
        self.model = YoloGoogleNet().to(device)
        self.criterion = YoloLoss().to(device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=0.0005
        )
        self.loss = []
        self.accuracy = []
        self.test_loss = []

    def set_args(self):
        """设置命令行参数"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_dir', type=str, default='/home/py/Comsen算法组/yolo/yolov1/dataset/images')
        parser.add_argument('--label_dir', type=str, default='/home/py/Comsen算法组/yolo/yolov1/dataset/labels')
        parser.add_argument('--save_dir', type=str, default='run/')
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--save_interval', help = '每10轮保持一轮', type=int, default=10)

        self.args = parser.parse_args()


    def train(self):
        """训练模型"""
        train_path_img = self.args.image_dir + '/train'
        train_path_label = self.args.label_dir + '/train'
        train_dataset = YoloDataset(train_path_img, train_path_label)

        val_path_img = self.args.image_dir + '/val'
        val_path_label = self.args.label_dir + '/val'
        val_dataset = YoloDataset(val_path_img, val_path_label)


        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
 
        # 创造保存目录
        os.makedirs(self.args.save_dir, exist_ok=True)


        for epoch in tqdm(range(self.args.epochs), desc = 'Epoch'):
            self.model.train()

            for image, label , _ in tqdm(train_loader, desc = 'Train', leave = False):
                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                self.loss.append(loss.item() / len(train_loader))

                del image, label


            # 验证
            self.model.eval()
            with torch.no_grad():
                for image, label , _ in tqdm(val_loader, desc = 'Val', leave = False):
                    image = image.to(self.device)
                    label = label.to(self.device)

                    output = self.model(image)
                    loss = self.criterion(output, label)

                    self.test_loss.append(loss.item() / len(val_loader))

                    del image, label

            # 保存模型
            if (epoch + 1) % self.args.save_interval == 0:
                save_path = os.path.join(self.args.save_dir, f'last.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f'Model saved to {save_path}')


        # 保存最终模型
        save_path = os.path.join(self.args.save_dir, f'final.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f'Final model saved to {save_path}')
            



            
if __name__ == '__main__':
    device = torch.device('cuda')
    trainer = trainer(device)
    trainer.train()









