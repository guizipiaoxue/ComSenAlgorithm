#!/usr/bin/env python3
"""删除指定范围的文件并重新编号后续文件

功能：
1. 删除 image/ 和 label/ 目录中序号从 525 到 540 的文件
2. 将序号 541 及之后的文件重新编号，向前移动以填补空缺

文件命名格式：
- image/XXXXXX.png (6位零填充)
- label/XXXXXX.json (6位零填充)

用法：
    python3 delete_and_renumber.py --start 525 --end 540 --dry-run  # 预览操作
    python3 delete_and_renumber.py --start 525 --end 540             # 执行删除和重命名
"""

import os
import argparse
from pathlib import Path


def delete_and_renumber(image_dir: str, label_dir: str, start_num: int, end_num: int, dry_run: bool = False):
    """删除指定范围的文件并重新编号后续文件
    
    Args:
        image_dir: 图片目录路径
        label_dir: 标签目录路径
        start_num: 起始序号（包含）
        end_num: 结束序号（包含）
        dry_run: 如果为 True，只显示将要执行的操作，不实际执行
    """
    if start_num > end_num:
        raise ValueError(f"起始序号 {start_num} 不能大于结束序号 {end_num}")
    
    # 要删除的文件数量
    delete_count = end_num - start_num + 1
    
    print(f"{'[预览模式]' if dry_run else '[执行模式]'}")
    print(f"将删除序号 {start_num:06d} 到 {end_num:06d} 的文件（共 {delete_count} 个）")
    print(f"将重新编号序号 {end_num + 1:06d} 及之后的文件\n")
    
    # 1. 删除指定范围的文件
    deleted_images = 0
    deleted_labels = 0
    
    print("=" * 60)
    print("第一步：删除指定范围的文件")
    print("=" * 60)
    
    for num in range(start_num, end_num + 1):
        # 删除图片
        image_file = os.path.join(image_dir, f"{num:06d}.png")
        if os.path.exists(image_file):
            print(f"删除图片: {image_file}")
            if not dry_run:
                os.remove(image_file)
            deleted_images += 1
        
        # 删除标签
        label_file = os.path.join(label_dir, f"{num:06d}.json")
        if os.path.exists(label_file):
            print(f"删除标签: {label_file}")
            if not dry_run:
                os.remove(label_file)
            deleted_labels += 1
    
    print(f"\n删除统计: 图片 {deleted_images} 个, 标签 {deleted_labels} 个\n")
    
    # 2. 收集需要重命名的文件
    print("=" * 60)
    print("第二步：重新编号后续文件")
    print("=" * 60)
    
    # 查找所有需要重命名的文件（序号 > end_num）
    image_files = []
    label_files = []
    
    if os.path.exists(image_dir):
        for file in sorted(os.listdir(image_dir)):
            if file.endswith('.png') and len(file) == 10:  # XXXXXX.png
                try:
                    num = int(file[:6])
                    if num > end_num:
                        image_files.append(num)
                except ValueError:
                    continue
    
    if os.path.exists(label_dir):
        for file in sorted(os.listdir(label_dir)):
            if file.endswith('.json') and len(file) == 11:  # XXXXXX.json
                try:
                    num = int(file[:6])
                    if num > end_num:
                        label_files.append(num)
                except ValueError:
                    continue
    
    # 3. 重命名文件
    renamed_images = 0
    renamed_labels = 0
    
    # 重命名图片
    for old_num in sorted(image_files):
        new_num = old_num - delete_count
        old_path = os.path.join(image_dir, f"{old_num:06d}.png")
        new_path = os.path.join(image_dir, f"{new_num:06d}.png")
        
        print(f"重命名图片: {old_num:06d}.png -> {new_num:06d}.png")
        if not dry_run:
            os.rename(old_path, new_path)
        renamed_images += 1
    
    # 重命名标签
    for old_num in sorted(label_files):
        new_num = old_num - delete_count
        old_path = os.path.join(label_dir, f"{old_num:06d}.json")
        new_path = os.path.join(label_dir, f"{new_num:06d}.json")
        
        print(f"重命名标签: {old_num:06d}.json -> {new_num:06d}.json")
        if not dry_run:
            os.rename(old_path, new_path)
        renamed_labels += 1
    
    print(f"\n重命名统计: 图片 {renamed_images} 个, 标签 {renamed_labels} 个")
    
    # 4. 输出总结
    print("\n" + "=" * 60)
    print("操作总结")
    print("=" * 60)
    print(f"删除的文件: 图片 {deleted_images} 个, 标签 {deleted_labels} 个")
    print(f"重命名的文件: 图片 {renamed_images} 个, 标签 {renamed_labels} 个")
    
    if dry_run:
        print("\n[这是预览模式，没有实际修改文件]")
        print("如需执行，请移除 --dry-run 参数")
    else:
        print("\n[操作已完成]")


def main():
    parser = argparse.ArgumentParser(
        description="删除指定范围的图片和标签文件，并重新编号后续文件"
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=525,
        help="起始序号（包含），默认 525"
    )
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=540,
        help="结束序号（包含），默认 540"
    )
    parser.add_argument(
        "--image-dir", "-i",
        type=str,
        default="./image",
        help="图片目录路径，默认 ./image"
    )
    parser.add_argument(
        "--label-dir", "-l",
        type=str,
        default="./label",
        help="标签目录路径，默认 ./label"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="预览模式：只显示将要执行的操作，不实际修改文件"
    )
    
    args = parser.parse_args()
    
    try:
        delete_and_renumber(
            args.image_dir,
            args.label_dir,
            args.start,
            args.end,
            args.dry_run
        )
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
