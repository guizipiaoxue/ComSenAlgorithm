import os
import shutil
import random
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def get_image_files(directory):
    """递归获取目录下所有图片文件"""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(Path(root) / file)
    return image_files

def main():
    parser = argparse.ArgumentParser(description="将数据集划分为训练集和验证集，并生成 YOLO 格式的目录结构和 yaml 文件")
    parser.add_argument("--image-dir", type=str, required=True, help="原始图片文件夹路径")
    parser.add_argument("--label-dir", type=str, required=True, help="原始标签文件夹路径 (YOLO 格式 .txt)")
    parser.add_argument("--output-dir", type=str, required=True, help="输出数据集路径")
    parser.add_argument("--split", type=float, default=0.8, help="训练集比例 (例如 0.8 表示 80% 训练集)")
    parser.add_argument("--classes", type=str, nargs='+', default=["object"], help="类别名称列表 (用于生成 yaml)")
    
    args = parser.parse_args()

    source_img_path = Path(args.image_dir)
    source_label_path = Path(args.label_dir)
    output_path = Path(args.output_dir)
    
    # 创建输出目录结构
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    train_label_dir = output_path / 'labels' / 'train'
    val_label_dir = output_path / 'labels' / 'val'
    
    for d in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 获取所有图片
    print(f"正在扫描图片: {source_img_path}")
    images = get_image_files(source_img_path)
    print(f"找到 {len(images)} 张图片")
    
    if not images:
        print("未找到图片，请检查路径")
        return

    # 随机打乱
    random.shuffle(images)
    
    # 划分
    train_count = int(len(images) * args.split)
    train_files = images[:train_count]
    val_files = images[train_count:]
    
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    
    # 复制文件函数
    def copy_files(files, dest_img_dir, dest_label_dir):
        missing_labels = 0
        for img_path in tqdm(files):
            # 复制图片
            shutil.copy2(img_path, dest_img_dir / img_path.name)
            
            # 寻找对应的标签文件
            label_name = img_path.stem + '.txt'
            label_src = source_label_path / label_name
            
            if label_src.exists():
                shutil.copy2(label_src, dest_label_dir / label_name)
            else:
                # 如果没有标签文件，创建一个空的
                (dest_label_dir / label_name).touch()
                missing_labels += 1
        return missing_labels

    print("正在处理训练集...")
    missing_train = copy_files(train_files, train_img_dir, train_label_dir)
    
    print("正在处理验证集...")
    missing_val = copy_files(val_files, val_img_dir, val_label_dir)
    
    if missing_train + missing_val > 0:
        print(f"警告: 共 {missing_train + missing_val} 张图片缺少对应的标签文件 (已生成空标签)")

    # 生成 data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(args.classes),
        'names': args.classes
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"完成！数据集已保存至: {output_path}")
    print(f"配置文件已生成: {yaml_path}")

if __name__ == "__main__":
    main()