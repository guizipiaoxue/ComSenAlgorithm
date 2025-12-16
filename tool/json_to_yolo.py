# 将 LabelMe JSON 格式转换为 YOLO 格式
# 注意这个文件，数量由图片的个数决定，如果没有读取到相对应的json，则用0 0 0 0 0 填充
# 要求提供图片路径，json路径，输出的路径

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_LINE = "0 0 0 0 0"

@dataclass
class ConversionStats:
	total_images: int = 0
	converted_with_json: int = 0
	missing_json: int = 0
	total_objects: int = 0

	def update_with_json(self, num_objects: int) -> None:
		self.converted_with_json += 1
		self.total_objects += num_objects

	def update_without_json(self) -> None:
		self.missing_json += 1


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="将 LabelMe JSON 标注转换为 YOLO txt 文件，确保每张图片都生成一个标签文件。",
	)
	parser.add_argument("images", help="图片文件夹路径")
	parser.add_argument("json_dir", help="LabelMe JSON 文件夹路径")
	parser.add_argument("output", help="输出 YOLO 标签文件夹路径")
	parser.add_argument(
		"--class-offset",
		type=int,
		default=1,
		help="LabelMe 中类别标签的起始编号，默认为 1（即 label '1' -> class_id 0）",
	)
	parser.add_argument(
		"--allow-empty-json",
		action="store_true",
		help="若 JSON 存在但未包含 shapes，则输出空文件而非默认占位行。",
	)
	return parser.parse_args()


def ensure_supported_image(path: Path) -> bool:
	return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(images_dir: Path) -> List[Path]:
	return sorted(p for p in images_dir.iterdir() if p.is_file() and ensure_supported_image(p))


def load_image_size(image_path: Path) -> Tuple[int, int]:
	img = cv2.imread(str(image_path))
	if img is not None:
		height, width = img.shape[:2]
		return width, height
	raise RuntimeError(
		"无法读取图片尺寸",
	)


def normalise_bbox(
	x_min: float,
	y_min: float,
	x_max: float,
	y_max: float,
	width: int,
	height: int,
) -> Tuple[float, float, float, float]:
	x_min = max(0.0, min(x_min, float(width)))
	y_min = max(0.0, min(y_min, float(height)))
	x_max = max(0.0, min(x_max, float(width)))
	y_max = max(0.0, min(y_max, float(height)))

	box_width = max(0.0, x_max - x_min)
	box_height = max(0.0, y_max - y_min)
	if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
		return 0.0, 0.0, 0.0, 0.0

	x_center = (x_min + x_max) / 2.0 / width
	y_center = (y_min + y_max) / 2.0 / height
	w_norm = box_width / width
	h_norm = box_height / height

	return (
		max(0.0, min(1.0, x_center)),
		max(0.0, min(1.0, y_center)),
		max(0.0, min(1.0, w_norm)),
		max(0.0, min(1.0, h_norm)),
	)


def parse_class_id(label: str, class_offset: int) -> int:
	try:
		class_id = int(label) - class_offset
		return class_id if class_id >= 0 else 0
	except (TypeError, ValueError):
		return 0


def iter_bboxes(shapes: Sequence[dict]) -> Iterable[Tuple[str, Sequence[Sequence[float]]]]:
	for shape in shapes:
		points = shape.get("points")
		if not points or len(points) < 2:
			continue
		label = shape.get("label", "0")
		yield label, points


def to_yolo_lines(
	shapes: Sequence[dict],
	width: int,
	height: int,
	class_offset: int,
) -> List[str]:
	yolo_lines: List[str] = []
	for label, points in iter_bboxes(shapes):
		xs = [float(p[0]) for p in points]
		ys = [float(p[1]) for p in points]
		x_min, x_max = min(xs), max(xs)
		y_min, y_max = min(ys), max(ys)
		x_center, y_center, w_norm, h_norm = normalise_bbox(x_min, y_min, x_max, y_max, width, height)
		if w_norm <= 0.0 or h_norm <= 0.0:
			continue
		class_id = parse_class_id(label, class_offset)
		yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
	return yolo_lines


def convert_single_image(
	image_path: Path,
	json_dir: Path,
	output_dir: Path,
	class_offset: int,
	allow_empty_json: bool,
	stats: ConversionStats,
) -> None:
	stats.total_images += 1

	width, height = load_image_size(image_path)
	json_path = json_dir / f"{image_path.stem}.json"
	output_path = output_dir / f"{image_path.stem}.txt"

	json_exists = json_path.exists()
	shapes = []

	if json_exists:
		try:
			with json_path.open("r", encoding="utf-8") as jf:
				data = json.load(jf)
			shapes = data.get("shapes", [])
		except Exception as exc:  # pragma: no cover - IO/JSON 异常
			print(f"读取 {json_path.name} 出错：{exc}，使用默认占位。")
			shapes = []

	yolo_lines = to_yolo_lines(shapes, width, height, class_offset)

	if yolo_lines:
		output_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
		stats.update_with_json(len(yolo_lines))
		return

	if json_exists:
		if allow_empty_json:
			output_path.write_text("", encoding="utf-8")
		else:
			output_path.write_text(DEFAULT_LINE + "\n", encoding="utf-8")
		stats.update_with_json(0)
		return

	stats.update_without_json()
	output_path.write_text(DEFAULT_LINE + "\n", encoding="utf-8")


def convert_dataset(
	images_dir: Path,
	json_dir: Path,
	output_dir: Path,
	class_offset: int,
	allow_empty_json: bool,
) -> ConversionStats:
	if not images_dir.exists() or not images_dir.is_dir():
		raise FileNotFoundError(f"图片目录不存在：{images_dir}")
	if not json_dir.exists() or not json_dir.is_dir():
		raise FileNotFoundError(f"JSON 目录不存在：{json_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)

	image_files = collect_images(images_dir)
	if not image_files:
		raise FileNotFoundError(f"图片目录中没有找到支持的图片文件：{images_dir}")

	stats = ConversionStats()
	for image_path in image_files:
		convert_single_image(
			image_path,
			json_dir,
			output_dir,
			class_offset,
			allow_empty_json,
			stats,
		)
	return stats


def main() -> None:
	args = parse_args()
	images_dir = Path(args.images).expanduser().resolve()
	json_dir = Path(args.json_dir).expanduser().resolve()
	output_dir = Path(args.output).expanduser().resolve()

	stats = convert_dataset(
		images_dir,
		json_dir,
		output_dir,
		class_offset=args.class_offset,
		allow_empty_json=args.allow_empty_json,
	)

	print("转换完成：")
	print(f"  总图片数: {stats.total_images}")
	print(f"  成功匹配 JSON: {stats.converted_with_json}")
	print(f"  缺失 JSON: {stats.missing_json}")
	print(f"  总对象数: {stats.total_objects}")
	print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
	main()

