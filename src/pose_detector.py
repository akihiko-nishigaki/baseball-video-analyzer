"""MediaPipe PoseLandmarker を使った骨格検出エンジン（Tasks API対応）"""

import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

# モデルファイルのパス
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "pose_landmarker_heavy.task")

# MediaPipe の関節名マッピング（日本語）
LANDMARK_NAMES_JA = {
    0: "鼻", 1: "左目内", 2: "左目", 3: "左目外", 4: "右目内",
    5: "右目", 6: "右目外", 7: "左耳", 8: "右耳", 9: "口左",
    10: "口右", 11: "左肩", 12: "右肩", 13: "左肘", 14: "右肘",
    15: "左手首", 16: "右手首", 17: "左小指", 18: "右小指",
    19: "左人差指", 20: "右人差指", 21: "左親指", 22: "右親指",
    23: "左腰", 24: "右腰", 25: "左膝", 26: "右膝",
    27: "左足首", 28: "右足首", 29: "左かかと", 30: "右かかと",
    31: "左つま先", 32: "右つま先",
}

# 野球分析で重要な関節インデックス
KEY_LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_index": 19, "right_index": 20,
    "nose": 0,
}

# 骨格の接続線定義（描画用）
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # 上半身
    (11, 23), (12, 24), (23, 24),  # 胴体
    (23, 25), (25, 27), (24, 26), (26, 28),  # 下半身
    (15, 19), (16, 20),  # 手首→指先
]


class PoseDetector:
    """MediaPipe PoseLandmarker による骨格検出クラス"""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 model_path=None):
        model = model_path or MODEL_PATH

        options = vision.PoseLandmarkerOptions(
            base_options=base_options_module.BaseOptions(
                model_asset_path=model,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        """1フレームから骨格を検出

        Args:
            frame: BGR画像 (numpy array)

        Returns:
            landmarks: 33個の関節座標リスト [(x, y, z, visibility), ...]
                       検出失敗時は None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks = []
        for lm in result.pose_landmarks[0]:
            landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
        return landmarks

    def draw_skeleton(self, frame, landmarks, show_angles=None):
        """骨格をフレームに描画（後方互換用ラッパー）"""
        return draw_skeleton(frame, landmarks, show_angles)

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def draw_skeleton(frame, landmarks, show_angles=None):
    """骨格をフレームに描画（PoseDetectorインスタンス不要）

    Args:
        frame: BGR画像
        landmarks: detect() の戻り値
        show_angles: 表示する角度のdict {名前: (idx_a, idx_b, idx_c)}

    Returns:
        描画済みフレーム
    """
    if landmarks is None:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 関節の座標をピクセルに変換
    points = {}
    for i, (x, y, z, v) in enumerate(landmarks):
        if v > 0.5:
            px, py = int(x * w), int(y * h)
            points[i] = (px, py)

    # 接続線を描画
    for start, end in SKELETON_CONNECTIONS:
        if start in points and end in points:
            cv2.line(overlay, points[start], points[end],
                     (0, 255, 0), 2, cv2.LINE_AA)

    # 関節点を描画
    for idx, (px, py) in points.items():
        if idx in KEY_LANDMARKS.values():
            cv2.circle(overlay, (px, py), 5, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (px, py), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # 角度テキストを描画
    if show_angles:
        from .angle_analyzer import calc_angle
        for name, (a, b, c) in show_angles.items():
            if a in points and b in points and c in points:
                angle = calc_angle(points[a], points[b], points[c])
                bx, by = points[b]
                cv2.putText(overlay, f"{angle:.0f}", (bx + 10, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.putText(overlay, f"{angle:.0f}", (bx + 10, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
                            cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    return result
