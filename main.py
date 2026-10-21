import argparse
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# [SỬA] Đường dẫn tính theo vị trí file main.py thay vì thư mục đang đứng,
# để chạy `python main.py` từ thư mục nào cũng tìm được model.
ROOT = Path(__file__).resolve().parent

# CONFIG
# ===============================

MODEL_PATH = ROOT / "models" / "best.pt"
VIDEO_PATH = ROOT / "videos" / "test.mp4"
# [SỬA] README nói kết quả được lưu ở output_images/ nhưng code cũ chỉ imshow, không lưu gì.
OUTPUT_DIR = ROOT / "output_images"

WINDOW_SIZE = 30        # frames

# [SỬA] Code cũ khai báo HIGH_RISK_FRAMES nhưng không dùng. Giờ nó là số frame (trong
# WINDOW_SIZE frame gần nhất) mà model phải dự đoán "drowning" thì mới báo HIGH_RISK.
HIGH_RISK_FRAMES = 15   # threshold
# [SỬA] Mới: ngưỡng thấp hơn cho mức cảnh báo POSSIBLE.
POSSIBLE_FRAMES = 5

# [SỬA] Mới: bỏ các detection có độ tin cậy thấp (code cũ dùng mặc định 0.25 của Ultralytics).
CONF_THRESHOLD = 0.4

# [SỬA] Mới: ngưỡng "gần như bất động", tính bằng (quãng dịch chuyển tâm bbox / chiều cao bbox)
# mỗi frame. Code cũ dùng pixel (2px, 5px) nên phụ thuộc độ phân giải và khoảng cách camera.
MOTIONLESS_THRESHOLD = 0.02

# [SỬA] Mới: xoá bộ nhớ của các ID đã mất dấu quá số frame này. Code cũ giữ mãi -> rò bộ nhớ
# khi chạy camera lâu.
LOST_TRACK_FRAMES = 60

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COLOR_MAP = {
    "NORMAL_ACTIVITY": (0, 255, 0),
    "POSSIBLE_DROWNING": (0, 255, 255),
    "HIGH_RISK_DROWNING": (0, 0, 255),
}


# [SỬA] Mới: cho phép đổi model/nguồn video/ngưỡng qua dòng lệnh thay vì phải sửa code.
def parse_args():
    parser = argparse.ArgumentParser(description="Drowning detection with YOLOv11 + temporal analysis")
    parser.add_argument("--model", default=str(MODEL_PATH), help="đường dẫn file .pt")
    parser.add_argument("--source", default=str(VIDEO_PATH),
                        help="video, ảnh, hoặc chỉ số webcam (vd: 0)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="ngưỡng confidence")
    parser.add_argument("--no-show", action="store_true", help="không mở cửa sổ hiển thị")
    parser.add_argument("--no-save", action="store_true", help="không lưu kết quả ra output_images/")
    return parser.parse_args()


# DISTANCE FUNCTION
# ===============================

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# [SỬA] Mới: dịch chuyển trung bình mỗi frame, chia cho chiều cao bbox để chuẩn hoá.
def avg_normalized_motion(track):
    movement = 0
    for i in range(1, len(track)):
        prev, cur = track[i-1], track[i]
        movement += distance(prev["center"], cur["center"]) / max(cur["height"], 1)
    return movement / (len(track) - 1)


# RISK ANALYSIS
# ===============================

def analyze_behavior(track):
    if len(track) < 5:
        return "NORMAL_ACTIVITY"

    # [SỬA] Code cũ chỉ dựa vào tốc độ di chuyển và BỎ QUA nhãn model dự đoán, nên người đứng
    # yên/thả nổi bị báo HIGH_RISK còn người đang vùng vẫy (di chuyển nhiều) lại là NORMAL.
    # Giờ quyết định chính dựa trên số frame model dự đoán "drowning" trong cửa sổ thời gian,
    # nên một frame nhận nhầm (lặn, quay người) không gây báo động.
    drowning_frames = sum(1 for s in track if s["drowning"])

    if drowning_frames >= HIGH_RISK_FRAMES:
        return "HIGH_RISK_DROWNING"

    if drowning_frames >= POSSIBLE_FRAMES:
        return "POSSIBLE_DROWNING"

    # [SỬA] Mới (model 3 lớp): người được nhận là "Person out of water" trong phần lớn cửa sổ
    # thì đang ở trên bờ, không thể đuối nước -> bỏ qua quy tắc bất động bên dưới. Nếu không,
    # người đứng yên trên thành hồ sẽ bị báo POSSIBLE nhầm.
    out_of_water_frames = sum(1 for s in track if s["out_of_water"])
    if out_of_water_frames > len(track) // 2:
        return "NORMAL_ACTIVITY"

    # [SỬA] Chuyển động giờ chỉ là tín hiệu phụ: bất động suốt cả cửa sổ (có thể là đuối nước
    # thụ động, nằm im dưới nước) thì chỉ cảnh báo mức POSSIBLE, không lên HIGH_RISK.
    if len(track) == WINDOW_SIZE and avg_normalized_motion(track) < MOTIONLESS_THRESHOLD:
        return "POSSIBLE_DROWNING"

    return "NORMAL_ACTIVITY"


# [SỬA] Mới: lấy chỉ số lớp "drowning" từ chính model thay vì hard-code, phòng khi train lại
# với thứ tự lớp khác.
def find_drowning_class(model):
    for idx, name in model.names.items():
        if name.lower() == "drowning":
            return idx
    raise ValueError(f"Model không có lớp 'drowning': {model.names}")


# [SỬA] Mới: tìm lớp "Person out of water" (chỉ có ở model 3 lớp train từ dataset
# "Swimming and Drowning Detection"). Model 2 lớp cũ không có -> trả về None.
def find_out_of_water_class(model):
    for idx, name in model.names.items():
        if "out of water" in name.lower():
            return idx
    return None


# [SỬA] Mới: README có nhắc chạy trên ảnh nhưng code cũ chỉ hỗ trợ video.
def run_image(model, source, args):
    r = model.predict(source=source, conf=args.conf, verbose=False)[0]
    frame = r.plot()

    if not args.no_save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / f"{Path(source).stem}_result.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"Saved: {out_path}")

    if not args.no_show:
        cv2.imshow("Drowning Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# VIDEO PIPELINE
# ===============================

def run_video(model, source, args):
    drowning_cls = find_drowning_class(model)
    out_of_water_cls = find_out_of_water_class(model)

    # [SỬA] Code cũ mở cv2.VideoCapture nhưng không dùng (model.track tự đọc video). Giờ chỉ mở
    # để kiểm tra nguồn và lấy FPS cho file video đầu ra.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    print("✅ Video loaded")

    # MEMORY FOR TRACKING
    # ===============================

    # [SỬA] Mỗi phần tử giờ lưu cả nhãn model (drowning hay không) + tâm + chiều cao bbox,
    # thay vì chỉ (cx, cy) như trước.
    memory = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    last_seen = {}
    alerted = set()
    writer = None

    # Built-in YOLO tracking
    results = model.track(source=source, stream=True, persist=True, conf=args.conf, verbose=False)

    for frame_idx, r in enumerate(results):

        frame = r.orig_img
        boxes = r.boxes

        # [SỬA] Code cũ `continue` khi frame không có box/ID -> bỏ qua luôn imshow, video bị
        # đứng hình mỗi khi không thấy người. Giờ chỉ bỏ qua phần vẽ, frame vẫn được hiển thị/lưu.
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for box, pid, cls, conf in zip(xyxy, ids, classes, confs):

                x1, y1, x2, y2 = map(int, box)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                memory[pid].append({
                    "drowning": cls == drowning_cls,
                    # [SỬA] Mới: luôn False với model 2 lớp (out_of_water_cls = None).
                    "out_of_water": cls == out_of_water_cls,
                    "center": (cx, cy),
                    "height": y2 - y1,
                })
                last_seen[pid] = frame_idx

                state = analyze_behavior(memory[pid])

                # [SỬA] Mới: in cảnh báo ra console một lần khi một ID chuyển sang HIGH_RISK.
                if state == "HIGH_RISK_DROWNING":
                    if pid not in alerted:
                        alerted.add(pid)
                        print(f"🚨 ALERT frame {frame_idx}: ID {pid} HIGH_RISK_DROWNING")
                else:
                    alerted.discard(pid)

                # ===============================
                # DRAW
                # ===============================

                # [SỬA] color_map chuyển ra hằng COLOR_MAP ở đầu file, không tạo lại mỗi vòng lặp.
                color = COLOR_MAP.get(state, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # [SỬA] Hiển thị thêm nhãn + confidence của model để dễ debug.
                cv2.putText(
                    frame,
                    f"ID {pid}: {state} ({model.names[cls]} {conf:.2f})",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # [SỬA] Mới: dọn bộ nhớ của các ID đã mất dấu lâu.
        for pid in [p for p, seen in last_seen.items() if frame_idx - seen > LOST_TRACK_FRAMES]:
            memory.pop(pid, None)
            last_seen.pop(pid)
            alerted.discard(pid)

        # [SỬA] Mới: lưu video kết quả.
        if not args.no_save:
            if writer is None:
                OUTPUT_DIR.mkdir(exist_ok=True)
                name = f"webcam{source}" if isinstance(source, int) else Path(source).stem
                out_path = OUTPUT_DIR / f"{name}_result.mp4"
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                print(f"Saving to: {out_path}")
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("Drowning Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # LOAD MODEL
    # ===============================

    # [SỬA] Code cũ load model ở cấp module (chạy ngay khi import). Giờ đưa vào main().
    model = YOLO(args.model)

    # [SỬA] Mới: nhận chỉ số webcam (vd "0") và báo lỗi rõ ràng khi file không tồn tại.
    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        print(f"❌ Source not found: {source}  (dùng --source để chỉ định video/ảnh)")
        return

    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS:
        run_image(model, source, args)
    else:
        run_video(model, source, args)


if __name__ == "__main__":
    main()
