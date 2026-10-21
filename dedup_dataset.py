"""Loại các ảnh trong valid/test gần như trùng với ảnh trong train.

Ảnh trùng với train làm mAP/recall trên valid/test cao hơn thực tế, vì model đã "thấy" ảnh đó
khi train. Script so sánh bằng difference hash (dHash) 16x16 = 256 bit và khoảng cách Hamming;
mỗi ảnh valid/test được so cả bản gốc lẫn bản lật ngang/dọc, vì train của Roboflow có augment lật.

Mặc định chỉ in báo cáo (dry run). Thêm --apply để CHUYỂN (không xoá) ảnh + nhãn trùng sang
<dataset>/_removed_duplicates/<split>/, muốn khôi phục thì chuyển ngược lại.

Ví dụ:
    python dedup_dataset.py "D:/.../Swimming and Drowning Detection.v1i.yolov11"
    python dedup_dataset.py "D:/.../Swimming and Drowning Detection.v1i.yolov11" --apply
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIZE = 16          # dHash 16x16 = 256 bit
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_SPLIT = "train"
EVAL_SPLITS = ["valid", "val", "test"]   # Roboflow dùng "valid", Ultralytics hay dùng "val"
CHUNK = 512             # số hash so sánh mỗi lần, để ma trận khoảng cách không quá lớn


def parse_args():
    parser = argparse.ArgumentParser(description="Loại ảnh valid/test trùng với train")
    parser.add_argument("dataset", help="thư mục dataset (chứa train/, valid/, test/)")
    parser.add_argument("--threshold", type=int, default=30,
                        help="khoảng cách Hamming tối đa (trên 256 bit) để coi là trùng")
    parser.add_argument("--apply", action="store_true",
                        help="chuyển ảnh trùng sang _removed_duplicates/ (mặc định chỉ báo cáo)")
    return parser.parse_args()


def dhash(img):
    # So sánh độ sáng 2 pixel cạnh nhau -> bền với thay đổi độ sáng/exposure của augment.
    a = np.asarray(img.resize((HASH_SIZE + 1, HASH_SIZE), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hash_variants(path, with_flips):
    img = Image.open(path).convert("L")
    variants = [img]
    if with_flips:
        variants += [
            img.transpose(Image.FLIP_LEFT_RIGHT),
            img.transpose(Image.FLIP_TOP_BOTTOM),
            img.transpose(Image.ROTATE_180),
        ]
    return [dhash(v) for v in variants]


def list_images(split_dir):
    return sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)


def to_signed(hashes):
    # Bit {0,1} -> {-1,+1}: khoảng cách Hamming = (số bit - tích vô hướng) / 2,
    # nên tính được cả ma trận khoảng cách bằng một phép nhân ma trận.
    return np.asarray(hashes, dtype=np.float32) * 2 - 1


def find_duplicates(train_hashes, eval_paths, threshold):
    train = to_signed(train_hashes)          # (N, 256)
    n_bits = train.shape[1]

    # Mỗi ảnh eval có 4 hash (gốc + 3 kiểu lật); owner[i] = chỉ số ảnh eval của hash thứ i.
    queries, owner = [], []
    for i, p in enumerate(eval_paths):
        for h in hash_variants(p, with_flips=True):
            queries.append(h)
            owner.append(i)
    queries = to_signed(queries)
    owner = np.asarray(owner)

    best_dist = np.full(len(eval_paths), n_bits + 1)
    best_match = np.full(len(eval_paths), -1)
    for start in range(0, len(queries), CHUNK):
        q = queries[start:start + CHUNK]
        dist = (n_bits - q @ train.T) / 2    # (chunk, N)
        idx = dist.argmin(axis=1)
        d = dist[np.arange(len(q)), idx]
        for o, di, ti in zip(owner[start:start + CHUNK], d, idx):
            if di < best_dist[o]:
                best_dist[o], best_match[o] = di, ti

    return [(i, int(best_match[i]), int(best_dist[i]))
            for i in range(len(eval_paths)) if best_dist[i] <= threshold]


def move_pair(img_path, split_dir, removed_dir):
    label_path = split_dir / "labels" / f"{img_path.stem}.txt"
    (removed_dir / "images").mkdir(parents=True, exist_ok=True)
    (removed_dir / "labels").mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_path), removed_dir / "images" / img_path.name)
    if label_path.exists():
        shutil.move(str(label_path), removed_dir / "labels" / label_path.name)


def main():
    args = parse_args()
    root = Path(args.dataset).resolve()
    removed_root = root / "_removed_duplicates"

    train_paths = list_images(root / TRAIN_SPLIT)
    print(f"Hashing {len(train_paths)} ảnh train...")
    train_hashes = [hash_variants(p, with_flips=False)[0] for p in train_paths]

    report_rows = []
    for split in EVAL_SPLITS:
        split_dir = root / split
        if not (split_dir / "images").is_dir():
            continue

        eval_paths = list_images(split_dir)
        dups = find_duplicates(train_hashes, eval_paths, args.threshold)
        print(f"{split}: {len(dups)}/{len(eval_paths)} ảnh trùng với train "
              f"({len(dups) / max(len(eval_paths), 1):.1%}), threshold={args.threshold}")

        for i, ti, d in dups:
            report_rows.append([split, eval_paths[i].name, train_paths[ti].name, d])
            if args.apply:
                move_pair(eval_paths[i], split_dir, removed_root / split)

        # Ultralytics cache danh sách nhãn trong labels.cache; xoá để lần train sau đọc lại.
        cache = split_dir / "labels.cache"
        if args.apply and dups and cache.exists():
            cache.unlink()

    report_path = root / "duplicates_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "file", "train_match", "hamming_distance"])
        writer.writerows(report_rows)
    print(f"Báo cáo: {report_path}")

    if args.apply:
        print(f"Đã chuyển {len(report_rows)} ảnh + nhãn sang {removed_root}")
    else:
        print("Dry run: chưa chuyển file nào. Chạy lại với --apply để loại ảnh trùng.")


if __name__ == "__main__":
    main()
