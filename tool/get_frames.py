
import argparse
import os
import sys
import cv2

def time_check(fps_sec, fps):
    # 在2分57秒到3分39秒之间，4分37秒到4分49秒之间，6分18秒到6分40秒之间
    if fps <= 0:
        raise ValueError("FPS 必须大于 0")
    if fps_sec >= (2 * 60 + 57) * fps and fps_sec < (3 * 60 + 39) * fps:
        return True
    if fps_sec >= (4 * 60 + 37) * fps and fps_sec < (4 * 60 + 49) * fps:
        return True
    if fps_sec >= (6 * 60 + 18) * fps and fps_sec < (6 * 60 + 40) * fps:
        return True
    return False


    if time_sec < 0:
        raise ValueError("时间秒数不能为负值")
    return time_sec
def extract_all_frames(input_path: str, output_dir: str,):
    """
    从视频中顺序提取每一帧并保存为图片
    
    Args:
        input_path: 输入视频文件路径
        output_dir: 输出图片目录
    Returns:
        保存的帧数
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入视频不存在: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_path}")

    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / video_fps) if video_fps > 0 else 0.0

    print(f"打开视频: {input_path}")
    print(f"FPS: {video_fps:.3f}, 总帧数: {total_frames}, 时长: {duration:.2f}s")
    print(f"输出目录: {output_dir}")
    print("-" * 60)

    saved = 0
    frame_idx = 0

    # 顺序读取每一帧
    while True:
        ret, frame = cap.read()
        
        # 检查是否成功读取
        if not ret or frame is None:
            print(f"\n在帧 {frame_idx} 处无法读取帧（可能已到视频末尾）。")
            break

        if time_check(frame_idx, video_fps) and frame_idx % 3 == 0:
            # 保存当前帧
            saved += 1
            file_name = f"{saved:06d}.png"
            out_path = os.path.join(output_dir, file_name)
            cv2.imwrite(out_path, frame)
            print(f"保存 {saved}: {out_path} (frame {saved})")
        frame_idx += 1

    cap.release()
    print("-" * 60)
    print(f"完成：共保存 {saved} 帧。")
    print(f"输出目录：{output_dir}")
    
    return saved


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(
        description="从视频中顺序提取每一帧并保存为图片（文件名为 6 位零填充）。"
    )
    p.add_argument(
        "--input", "-i", 
        default="test_video.mp4", 
        help="输入视频文件路径（默认：test_video.mp4）"
    )
    p.add_argument(
        "--output", "-o", 
        default="./drone_img/", 
        help="输出图片目录（默认：./drone_img/）"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        saved = extract_all_frames(
            args.input, 
            args.output, 
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
