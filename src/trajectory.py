"""バット軌道・手首軌跡の描画と分析"""

import cv2
import numpy as np


def draw_wrist_trajectory(frame, landmarks_history, current_frame,
                          trail_length=40, wrist_indices=(15, 16)):
    """手首の軌跡をフレーム上に描画

    Args:
        frame: BGR画像
        landmarks_history: 全フレームのlandmarks
        current_frame: 現在のフレーム番号
        trail_length: 軌跡の長さ（フレーム数）
        wrist_indices: 描画する手首インデックス

    Returns:
        描画済みフレーム
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    trail_start = max(0, current_frame - trail_length)

    for wrist_idx in wrist_indices:
        # 色設定: 右手首=シアン, 左手首=マゼンタ
        base_color = (255, 255, 0) if wrist_idx == 16 else (255, 0, 255)

        points = []
        for f in range(trail_start, current_frame + 1):
            lm = landmarks_history.get(f)
            if lm and lm[wrist_idx][3] > 0.5:
                px = int(lm[wrist_idx][0] * w)
                py = int(lm[wrist_idx][1] * h)
                points.append((px, py))

        # 軌跡を線で描画（グラデーション）
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            color = tuple(int(c * alpha) for c in base_color)
            cv2.line(overlay, points[i-1], points[i], color, thickness, cv2.LINE_AA)

        # 現在位置に大きな点
        if points:
            cv2.circle(overlay, points[-1], 6, base_color, -1, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    return result


def draw_bat_path(frame, landmarks_history, current_frame, trail_length=30):
    """バットの軌道を推定して描画

    バットの先端は手首と指先の延長線上にあると推定。

    Args:
        frame: BGR画像
        landmarks_history: 全フレームのlandmarks
        current_frame: 現在のフレーム番号
        trail_length: 軌跡の長さ

    Returns:
        描画済みフレーム
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    trail_start = max(0, current_frame - trail_length)

    bat_tips = []
    for f in range(trail_start, current_frame + 1):
        lm = landmarks_history.get(f)
        if lm is None:
            bat_tips.append(None)
            continue

        # 右手首(16)と右人差し指(20)を使ってバット先端を推定
        rw = lm[16]  # 右手首
        ri = lm[20]  # 右人差し指

        if rw[3] > 0.5 and ri[3] > 0.5:
            # 手首→指先の方向に延長（バットの長さ分）
            dx = ri[0] - rw[0]
            dy = ri[1] - rw[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0.001:
                # バットの長さ = 手首→指先の約3倍と仮定
                scale = 3.0
                bat_x = rw[0] + dx * scale
                bat_y = rw[1] + dy * scale
                bat_tips.append((int(bat_x * w), int(bat_y * h)))
            else:
                bat_tips.append(None)
        else:
            bat_tips.append(None)

    # バット軌道を描画
    valid_tips = [(i, p) for i, p in enumerate(bat_tips) if p is not None]
    for j in range(1, len(valid_tips)):
        idx, pt = valid_tips[j]
        _, prev_pt = valid_tips[j-1]
        alpha = (j + 1) / len(valid_tips)
        color = (0, int(200 * alpha), int(255 * alpha))
        thickness = max(1, int(4 * alpha))
        cv2.line(overlay, prev_pt, pt, color, thickness, cv2.LINE_AA)

    # 現在のバット位置
    if valid_tips:
        _, last_pt = valid_tips[-1]
        cv2.circle(overlay, last_pt, 8, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, last_pt, 8, (255, 255, 255), 2, cv2.LINE_AA)

    # 現フレームのバットの線（手首→バット先端）
    lm = landmarks_history.get(current_frame)
    if lm and lm[16][3] > 0.5 and bat_tips and bat_tips[-1]:
        wrist_px = (int(lm[16][0] * w), int(lm[16][1] * h))
        cv2.line(overlay, wrist_px, bat_tips[-1], (255, 255, 255), 2, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    return result


def draw_phase_indicator(frame, phase_key, phase_info, progress_ratio=0):
    """フェーズ表示バナーを描画

    Args:
        frame: BGR画像
        phase_key: フェーズキー
        phase_info: BATTING_PHASES の値 dict
        progress_ratio: フェーズ内の進行率 (0-1)

    Returns:
        描画済みフレーム
    """
    h, w = frame.shape[:2]

    # フェーズ名の背景バー
    bar_height = 40
    bar_y = 10

    # 背景（半透明）
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, bar_y), (w - 10, bar_y + bar_height),
                  (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # プログレスバー
    if progress_ratio > 0:
        prog_width = int((w - 20) * progress_ratio)
        # フェーズの色をBGRに変換
        hex_color = phase_info["color"].lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        cv2.rectangle(frame, (10, bar_y), (10 + prog_width, bar_y + bar_height),
                      (b, g, r), -1)

    # テキスト
    text = f"{phase_info['emoji']} {phase_info['name']}"
    cv2.putText(frame, text, (20, bar_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def calc_swing_arc_angle(landmarks_history, swing):
    """スイングアーク角度を計算

    バット軌道が水平に対してどれだけの角度かを計算。
    レベルスイング（水平）= 0°、ダウンスイング = 負、アッパースイング = 正

    Args:
        landmarks_history: 全フレームのlandmarks
        swing: (start, end, peak, peak_speed)

    Returns:
        arc_angle: 角度（度）、計算不能時は None
    """
    start, end, peak, _ = swing

    # スイング開始付近と終了付近の手首位置
    start_lm = landmarks_history.get(start)
    end_lm = landmarks_history.get(min(peak + 3, end))

    if start_lm is None or end_lm is None:
        return None

    rw_start = start_lm[16]
    rw_end = end_lm[16]

    if rw_start[3] < 0.5 or rw_end[3] < 0.5:
        return None

    dx = rw_end[0] - rw_start[0]
    dy = rw_end[1] - rw_start[1]

    # Y軸は下向きなので反転
    arc_angle = -np.degrees(np.arctan2(dy, abs(dx)))
    return arc_angle
