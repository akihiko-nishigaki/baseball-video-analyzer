"""関節角度の計算と分析"""

import numpy as np


def calc_angle(a, b, c):
    """3点から角度を計算（度）

    Args:
        a, b, c: (x, y) 座標タプル。b が頂点。

    Returns:
        角度（0-180度）
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosine = np.dot(ba, bc) / norm
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


# バッティングで見るべき角度定義
# {名前: (関節A, 頂点B, 関節C)} ※インデックスはMediaPipe準拠
BATTING_ANGLES = {
    "右肘": (12, 14, 16),   # 右肩 → 右肘 → 右手首
    "左肘": (11, 13, 15),   # 左肩 → 左肘 → 左手首
    "右膝": (24, 26, 28),   # 右腰 → 右膝 → 右足首
    "左膝": (23, 25, 27),   # 左腰 → 左膝 → 左足首
    "右肩": (14, 12, 24),   # 右肘 → 右肩 → 右腰
    "左肩": (13, 11, 23),   # 左肘 → 左肩 → 左腰
}

# ピッチングで見るべき角度定義
PITCHING_ANGLES = {
    "投げ腕肘": (12, 14, 16),  # 右投手想定
    "投げ腕肩": (14, 12, 24),
    "軸足膝": (24, 26, 28),
    "踏み出し膝": (23, 25, 27),
    "体幹": (12, 24, 26),      # 右肩 → 右腰 → 右膝
}


def analyze_frame_angles(landmarks, angle_defs, image_size=None):
    """1フレームの全角度を計算

    Args:
        landmarks: PoseDetector.detect() の戻り値
        angle_defs: 角度定義dict
        image_size: (width, height) ピクセル座標に変換する場合

    Returns:
        dict {角度名: 角度値}
    """
    if landmarks is None:
        return {}

    if image_size:
        w, h = image_size
        points = {i: (lm[0] * w, lm[1] * h) for i, lm in enumerate(landmarks)
                  if lm[3] > 0.5}
    else:
        points = {i: (lm[0], lm[1]) for i, lm in enumerate(landmarks)
                  if lm[3] > 0.5}

    results = {}
    for name, (a, b, c) in angle_defs.items():
        if a in points and b in points and c in points:
            results[name] = calc_angle(points[a], points[b], points[c])
    return results


def calc_body_rotation(landmarks):
    """体の開き具合を計算（両肩の角度）

    Returns:
        回旋角度（度）。0=完全に正面、90=完全に横向き
    """
    if landmarks is None:
        return None

    ls = landmarks[11]  # 左肩
    rs = landmarks[12]  # 右肩

    if ls[3] < 0.5 or rs[3] < 0.5:
        return None

    dx = rs[0] - ls[0]
    dy = rs[1] - ls[1]
    # 水平方向の肩の開き
    angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
    return angle


def calc_center_of_gravity(landmarks):
    """重心位置を推定（両腰の中点）

    Returns:
        (x, y) 正規化座標、または None
    """
    if landmarks is None:
        return None

    lh = landmarks[23]  # 左腰
    rh = landmarks[24]  # 右腰

    if lh[3] < 0.5 or rh[3] < 0.5:
        return None

    cx = (lh[0] + rh[0]) / 2
    cy = (lh[1] + rh[1]) / 2
    return (cx, cy)


def get_angle_color(angle, ideal_min, ideal_max):
    """角度が理想範囲内かを色で返す

    Returns:
        (B, G, R) タプル
    """
    if ideal_min <= angle <= ideal_max:
        return (76, 175, 80)    # 緑 - 良好
    elif abs(angle - ideal_min) <= 15 or abs(angle - ideal_max) <= 15:
        return (0, 193, 255)    # 黄 - 注意
    else:
        return (67, 67, 244)    # 赤 - 要改善
