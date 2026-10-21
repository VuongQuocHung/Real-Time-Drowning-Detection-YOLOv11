import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# [SỬA] Đường dẫn tính theo vị trí file train.py, chạy từ thư mục nào cũng được.
ROOT = Path(__file__).resolve().parent

# [SỬA] Code cũ dùng data="data.yaml" nhưng file thật nằm ở dataset/data.yaml -> lỗi không tìm thấy
# dataset. Phải dùng đường dẫn tuyệt đối: data.yaml không có khoá `path:`, nên Ultralytics lấy thư
# mục chứa yaml làm gốc; nếu đường dẫn là tương đối, nó lại bị ghép vào thư mục datasets mặc định.
DATA_YAML = ROOT / "dataset" / "data.yaml"


# [SỬA] Mới: chỉnh tham số qua dòng lệnh thay vì phải sửa code.
def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 drowning detector")
    parser.add_argument("--weights", default="yolo11n.pt", help="model khởi tạo (vd yolo11n.pt, yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="mặc định: 16 nếu GPU, 8 nếu CPU")
    parser.add_argument("--device", default=None, help="mặc định: tự chọn GPU nếu có, không thì CPU")
    parser.add_argument("--name", default="yolov11_binary")
    return parser.parse_args()


def main():
    args = parse_args()

    # [SỬA] Code cũ ép device="cpu" (để né lỗi CUDA), nên máy có GPU vẫn train bằng CPU rất chậm.
    # Giờ tự dùng GPU nếu có.
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    # [SỬA] batch=8 chỉ cần cho CPU; GPU dùng batch lớn hơn.
    batch = args.batch or (8 if device == "cpu" else 16)
    print(f"Device: {device}, batch: {batch}")

    # [SỬA] Dùng file yolo11n.pt có sẵn trong repo nếu có; nếu không thì để Ultralytics tự tải.
    weights = ROOT / args.weights
    model = YOLO(str(weights) if weights.exists() else args.weights)

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        optimizer="SGD",
        lr0=0.01,
        patience=15,
        device=device,
        # [SỬA] project dùng đường dẫn tuyệt đối để runs/ luôn nằm trong repo.
        project=str(ROOT / "runs" / "drowning"),
        name=args.name,
        verbose=True
    )

    # [SỬA] Mới: code cũ chỉ đánh giá trên tập val (dùng để chọn best.pt), nên số liệu bị lạc quan.
    # Giờ đánh giá thêm best.pt trên tập test.
    best_path = Path(model.trainer.best)
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        project=str(ROOT / "runs" / "drowning"),
        name=f"{args.name}_test",
    )

    print("\n===== TEST SET =====")
    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    for i, cls in enumerate(metrics.box.ap_class_index):
        p, r, ap50, _ = metrics.box.class_result(i)
        print(f"{best_model.names[int(cls)]:>10}: precision={p:.3f} recall={r:.3f} mAP50={ap50:.3f}")

    # [SỬA] Không tự ghi đè models/best.pt (model gốc của repo); in đường dẫn để tự copy nếu muốn.
    print(f"\nBest weights: {best_path}")
    print(f"Để main.py dùng model mới: copy file trên vào {ROOT / 'models' / 'best.pt'}"
          f" hoặc chạy `python main.py --model \"{best_path}\"`")


if __name__ == "__main__":
    main()
