"""YOLOv8-segによるバット物体検出・トラッキング（セグメンテーション方式）"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# COCO dataset class index for "baseball bat"
BASEBALL_BAT_CLASS = 38


class BatDetector:
    """YOLOv8-segを使ったバット検出クラス（セグメンテーション方式）"""

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
        # セグメンテーションモデルを使用
        self.model = YOLO(f"yolov8{model_size}-seg.pt")
        self.confidence = confidence

    def detect(self, frame, wrist_pos=None):
        """1フレームからバットを検出（セグメンテーション）

        Args:
            frame: BGR画像 (numpy array)
            wrist_pos: 右手首のピクセル座標 (x, y) - 先端特定に使用

        Returns:
            dict: {bbox, confidence, center, tip, handle, mask} or None
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

        # セグメンテーションマスクから先端を特定
        tip, handle = self._find_tip_from_mask(results[0], best_idx, frame.shape, wrist_pos)

        # マスクが取れなかった場合はbboxコーナーにフォールバック
        if tip is None:
            tip, handle = self._estimate_tip_from_bbox(box, wrist_pos)

        return {
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "confidence": float(conf),
            "center": (float(cx), float(cy)),
            "tip": tip,
            "handle": handle,
        }

    def _find_tip_from_mask(self, result, idx, frame_shape, wrist_pos):
        """セグメンテーションマスクの輪郭からバット先端を特定

        マスクの輪郭点のうち、手首から最も遠い点 = バット先端。
        手首から最も近い点 = グリップ端。

        Args:
            result: YOLO推論結果
            idx: 使用するマスクのインデックス
            frame_shape: フレームの(h, w, c)
            wrist_pos: (wx, wy) 手首座標

        Returns:
            (tip, handle) or (None, None)
        """
        if result.masks is None or len(result.masks) <= idx:
            return None, None

        # マスクの正規化座標を取得
        mask_xy = result.masks[idx].xy
        if len(mask_xy) == 0 or len(mask_xy[0]) < 3:
            return None, None

        contour = mask_xy[0]  # (N, 2) の輪郭点配列

        h, w = frame_shape[:2]

        if wrist_pos is None:
            # 手首情報なし: 輪郭の重心から最も遠い点を先端とする
            centroid = contour.mean(axis=0)
            dists = np.sqrt(np.sum((contour - centroid) ** 2, axis=1))
            tip_idx = np.argmax(dists)
            tip_pt = contour[tip_idx]

            # 先端の反対側（先端から最も遠い輪郭点）をグリップとする
            dists_from_tip = np.sqrt(np.sum((contour - tip_pt) ** 2, axis=1))
            handle_idx = np.argmax(dists_from_tip)
            handle_pt = contour[handle_idx]

            return (float(tip_pt[0]), float(tip_pt[1])), \
                   (float(handle_pt[0]), float(handle_pt[1]))

        # 手首から最も遠い輪郭点 = バット先端
        wx, wy = wrist_pos
        dists = np.sqrt((contour[:, 0] - wx) ** 2 + (contour[:, 1] - wy) ** 2)
        tip_idx = np.argmax(dists)
        tip_pt = contour[tip_idx]

        # 手首から最も近い輪郭点 = グリップ端
        handle_idx = np.argmin(dists)
        handle_pt = contour[handle_idx]

        return (float(tip_pt[0]), float(tip_pt[1])), \
               (float(handle_pt[0]), float(handle_pt[1]))

    def _estimate_tip_from_bbox(self, bbox, wrist_pos):
        """フォールバック: bboxコーナーからバット先端を推定"""
        x1, y1, x2, y2 = bbox

        corners = [
            (float(x1), float(y1)),
            (float(x2), float(y1)),
            (float(x1), float(y2)),
            (float(x2), float(y2)),
        ]

        if wrist_pos is None:
            return corners[3], corners[0]

        wx, wy = wrist_pos
        dists = [(c[0] - wx) ** 2 + (c[1] - wy) ** 2 for c in corners]
        handle_idx = int(np.argmin(dists))
        diagonal_map = {0: 3, 1: 2, 2: 1, 3: 0}
        tip_idx = diagonal_map[handle_idx]

        return corners[tip_idx], corners[handle_idx]

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
            frame = reader.get_frame(i)
            if frame is None:
                continue

            # 手首座標をピクセルに変換
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
