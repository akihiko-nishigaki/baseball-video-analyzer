"""YOLOv8によるバット物体検出・トラッキング"""

import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO dataset class index for "baseball bat"
BASEBALL_BAT_CLASS = 38


class BatDetector:
    """YOLOv8を使ったバット検出クラス"""

    def __init__(self, model_size="n", confidence=0.3):
        """
        Args:
            model_size: YOLOv8モデルサイズ ("n"=nano, "s"=small, "m"=medium)
            confidence: 検出信頼度の閾値
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics がインストールされていません。"
                "'pip install ultralytics' を実行してください。"
            )
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.confidence = confidence

    def detect(self, frame, wrist_pos=None):
        """1フレームからバットを検出

        Args:
            frame: BGR画像 (numpy array)
            wrist_pos: 右手首のピクセル座標 (x, y) - 先端推定に使用

        Returns:
            dict: {bbox, confidence, center, tip, handle} or None
                bbox: (x1, y1, x2, y2) バウンディングボックス
                confidence: 検出信頼度
                center: (cx, cy) バットの中心座標
                tip: (tx, ty) バット先端の推定座標
                handle: (hx, hy) グリップ側の推定座標
        """
        results = self.model(
            frame,
            classes=[BASEBALL_BAT_CLASS],
            conf=self.confidence,
            verbose=False,
        )

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        # 最も信頼度の高い検出を使用
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()

        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # バット先端の推定
        tip, handle = self._estimate_tip(box, wrist_pos)

        return {
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "confidence": float(conf),
            "center": (float(cx), float(cy)),
            "tip": tip,
            "handle": handle,
        }

    def _estimate_tip(self, bbox, wrist_pos=None):
        """バウンディングボックスからバット先端とグリップ端を推定

        bboxの4隅のうち、対角線上にある2コーナーをバットの両端とみなす。
        手首に最も近いコーナーがグリップ側、対角のコーナーが先端。

        Args:
            bbox: (x1, y1, x2, y2)
            wrist_pos: (wx, wy) 手首のピクセル座標

        Returns:
            (tip, handle): それぞれ (x, y) のタプル
        """
        x1, y1, x2, y2 = bbox

        # bboxの4コーナー
        corners = [
            (float(x1), float(y1)),  # 左上
            (float(x2), float(y1)),  # 右上
            (float(x1), float(y2)),  # 左下
            (float(x2), float(y2)),  # 右下
        ]

        if wrist_pos is None:
            # 手首情報なし: bbox中心から最も遠いコーナーを先端とする
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # 対角線が長い方を使用（左上-右下 vs 右上-左下）
            return corners[3], corners[0]

        # 手首に最も近いコーナー = グリップ側
        wx, wy = wrist_pos
        dists = [(c[0] - wx) ** 2 + (c[1] - wy) ** 2 for c in corners]
        handle_idx = int(np.argmin(dists))
        handle = corners[handle_idx]

        # 対角コーナー = 先端 (0↔3, 1↔2)
        diagonal_map = {0: 3, 1: 2, 2: 1, 3: 0}
        tip_idx = diagonal_map[handle_idx]
        tip = corners[tip_idx]

        return tip, handle

    def detect_all_frames(self, reader, landmarks_history=None, progress_cb=None):
        """全フレームでバット検出を実行

        Args:
            reader: VideoReaderインスタンス
            landmarks_history: {frame_idx: landmarks} MediaPipeの関節データ
            progress_cb: 進捗コールバック (current, total) -> None

        Returns:
            detections: {frame_idx: detection_result, ...}
        """
        detections = {}
        total = reader.total_frames

        for i in range(total):
            frame = reader.read_frame(i)
            if frame is None:
                continue

            # 手首座標をピクセルに変換（検出精度向上のため）
            wrist_pos = None
            if landmarks_history and i in landmarks_history:
                lm = landmarks_history[i]
                if lm and lm[16][3] > 0.5:
                    h, w = frame.shape[:2]
                    wrist_pos = (lm[16][0] * w, lm[16][1] * h)

            result = self.detect(frame, wrist_pos=wrist_pos)
            if result:
                detections[i] = result

            if progress_cb:
                progress_cb(i + 1, total)

        return detections
