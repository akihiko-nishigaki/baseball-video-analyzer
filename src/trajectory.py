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


def _estimate_bat_tip(lm):
    """1フレームのlandmarksからバット先端位置を推定（フォールバック用）

    Args:
        lm: landmarks リスト [(x, y, z, visibility), ...]

    Returns:
        (tip_x, tip_y) 正規化座標 or None
    """
    VIS = 0.25

    rw = lm[16]
    ri = lm[20]
    re = lm[14]
    lw = lm[15]

    if rw[3] < VIS:
        return None

    if lw[3] > VIS:
        grip_x = (rw[0] + lw[0]) / 2
        grip_y = (rw[1] + lw[1]) / 2
    else:
        grip_x, grip_y = rw[0], rw[1]

    if ri[3] > VIS:
        dx = ri[0] - rw[0]
        dy = ri[1] - rw[1]
        ref_len = np.sqrt(dx ** 2 + dy ** 2)
        if ref_len > 0.005:
            scale = 3.5
            tip_x = grip_x + (dx / ref_len) * ref_len * scale
            tip_y = grip_y + (dy / ref_len) * ref_len * scale
            return (tip_x, tip_y)

    if re[3] > VIS:
        dx = rw[0] - re[0]
        dy = rw[1] - re[1]
        ref_len = np.sqrt(dx ** 2 + dy ** 2)
        if ref_len > 0.005:
            scale = 2.5
            tip_x = grip_x + (dx / ref_len) * ref_len * scale
            tip_y = grip_y + (dy / ref_len) * ref_len * scale
            return (tip_x, tip_y)

    return None


def compute_motion_bat_tips(reader, landmarks_history, progress_cb=None):
    """フレーム差分でバット先端位置を検出

    連続フレーム間の差分から動きのある領域を検出し、
    手首から外側方向で最も遠い動き点をバット先端とする。

    Args:
        reader: VideoReader
        landmarks_history: {frame_idx: landmarks}
        progress_cb: callback(current, total)

    Returns:
        {frame_idx: (tip_x, tip_y)} ピクセル座標
    """
    tips = {}
    prev_gray = None
    total = reader.total_frames

    for frame_idx, frame in reader.iter_frames():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_gray is not None:
            lm = landmarks_history.get(frame_idx)
            if lm is not None:
                tip = _detect_tip_from_motion(prev_gray, gray, lm, frame.shape)
                if tip:
                    tips[frame_idx] = tip

        prev_gray = gray

        if progress_cb and frame_idx % 10 == 0:
            progress_cb(frame_idx, total)

    return tips


def _detect_tip_from_motion(prev_gray, curr_gray, lm, frame_shape):
    """1フレーム分のモーションベースのバット先端検出

    手首から外側方向（体の中心→手首→延長）に向かう
    動き領域の最遠点をバット先端とする。

    Args:
        prev_gray: 前フレームのグレースケール
        curr_gray: 現フレームのグレースケール
        lm: landmarks
        frame_shape: (h, w, c)

    Returns:
        (tip_x, tip_y) ピクセル座標 or None
    """
    VIS = 0.2
    h, w = frame_shape[:2]

    rw = lm[16]
    if rw[3] < VIS:
        return None

    wrist_x = int(rw[0] * w)
    wrist_y = int(rw[1] * h)

    # 体の中心点（腰 or 肩の中点）
    if lm[23][3] > VIS and lm[24][3] > VIS:
        cx = int((lm[23][0] + lm[24][0]) / 2 * w)
        cy = int((lm[23][1] + lm[24][1]) / 2 * h)
    elif lm[11][3] > VIS and lm[12][3] > VIS:
        cx = int((lm[11][0] + lm[12][0]) / 2 * w)
        cy = int((lm[11][1] + lm[12][1]) / 2 * h)
    else:
        return None

    # フレーム差分
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # ノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 動きのあるピクセル
    ys, xs = np.where(mask > 0)
    if len(ys) < 30:
        return None

    # 体→手首方向（外側方向）
    out_dx = float(wrist_x - cx)
    out_dy = float(wrist_y - cy)
    out_len = np.sqrt(out_dx ** 2 + out_dy ** 2)
    if out_len < 10:
        return None
    out_dx /= out_len
    out_dy /= out_len

    # 手首からの相対位置
    rel_x = xs.astype(np.float32) - wrist_x
    rel_y = ys.astype(np.float32) - wrist_y

    # 外側方向の成分が正（手首より外側にある点のみ）
    dots = rel_x * out_dx + rel_y * out_dy
    outward = dots > 0

    if not np.any(outward):
        return None

    out_xs = xs[outward]
    out_ys = ys[outward]
    dists = np.sqrt((out_xs - wrist_x) ** 2 + (out_ys - wrist_y) ** 2)

    # 手そのものを除外（最低距離）
    min_dist = max(30, out_len * 0.3)
    far_enough = dists > min_dist

    if not np.any(far_enough):
        return None

    far_dists = dists[far_enough]
    far_xs = out_xs[far_enough]
    far_ys = out_ys[far_enough]

    # 上位5%の点の重心をバット先端とする（外れ値ノイズ軽減）
    threshold = np.percentile(far_dists, 95)
    top = far_dists >= threshold

    tip_x = int(np.mean(far_xs[top]))
    tip_y = int(np.mean(far_ys[top]))

    return (tip_x, tip_y)


def draw_bat_path(frame, landmarks_history, current_frame, trail_length=30,
                  motion_tips=None):
    """バットの軌道を描画（モーション検出優先、推定フォールバック）

    Args:
        frame: BGR画像
        landmarks_history: 全フレームのlandmarks
        current_frame: 現在のフレーム番号
        trail_length: 軌跡の長さ
        motion_tips: {frame_idx: (x, y)} モーション検出結果（ピクセル座標）

    Returns:
        描画済みフレーム
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    trail_start = max(0, current_frame - trail_length)

    # 先端座標を収集（モーション検出 > 推定 の優先順位）
    raw_tips = []
    for f in range(trail_start, current_frame + 1):
        # モーション検出結果があればそちらを使う
        if motion_tips and f in motion_tips:
            raw_tips.append((f, motion_tips[f]))
            continue

        # フォールバック: MediaPipe推定
        lm = landmarks_history.get(f)
        if lm is None:
            continue
        tip = _estimate_bat_tip(lm)
        if tip:
            raw_tips.append((f, (int(tip[0] * w), int(tip[1] * h))))

    # 欠落フレームを線形補間で埋める
    tips = _interpolate_tips(raw_tips, max_gap=6)

    # バット軌道を描画
    for j in range(1, len(tips)):
        _, pt = tips[j]
        _, prev_pt = tips[j - 1]
        alpha = (j + 1) / len(tips)
        color = (0, int(200 * alpha), int(255 * alpha))
        thickness = max(1, int(4 * alpha))
        cv2.line(overlay, prev_pt, pt, color, thickness, cv2.LINE_AA)

    # 現在のバット位置マーカー
    if tips:
        _, last_pt = tips[-1]
        cv2.circle(overlay, last_pt, 8, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, last_pt, 8, (255, 255, 255), 2, cv2.LINE_AA)

    # 現フレームのバットの線（グリップ→先端）
    lm = landmarks_history.get(current_frame)
    cur_tip_px = None
    if motion_tips and current_frame in motion_tips:
        cur_tip_px = motion_tips[current_frame]
    elif lm:
        est = _estimate_bat_tip(lm)
        if est:
            cur_tip_px = (int(est[0] * w), int(est[1] * h))

    if lm and lm[16][3] > 0.25 and cur_tip_px:
        rw = lm[16]
        lw = lm[15]
        if lw[3] > 0.25:
            grip_px = (int((rw[0] + lw[0]) / 2 * w), int((rw[1] + lw[1]) / 2 * h))
        else:
            grip_px = (int(rw[0] * w), int(rw[1] * h))
        cv2.line(overlay, grip_px, cur_tip_px, (255, 255, 255), 2, cv2.LINE_AA)

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


def draw_ghost_skeletons(frame, landmarks_history, current_frame,
                         ghost_count=5, ghost_step=3):
    """残像（ゴースト）骨格を半透明で描画

    過去のフレームの骨格を薄い色で重ねて表示し、動きの軌跡を視覚化する。

    Args:
        frame: BGR画像
        landmarks_history: 全フレームのlandmarksデータ
        current_frame: 現在のフレーム番号
        ghost_count: 表示するゴーストの数 (default 5)
        ghost_step: ゴースト間のフレーム間隔 (default 3)

    Returns:
        描画済みフレーム
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 骨格接続線の定義
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (15, 19), (16, 20),
    ]

    for gi in range(ghost_count, 0, -1):
        ghost_frame = current_frame - gi * ghost_step
        if ghost_frame < 0:
            continue

        lm = landmarks_history.get(ghost_frame)
        if lm is None:
            continue

        # 古いほど透明度を上げる (alphaが小さい=薄い)
        alpha = (ghost_count - gi + 1) / (ghost_count + 1)
        color_intensity = int(255 * alpha * 0.6)
        line_color = (color_intensity, int(color_intensity * 0.7), 0)  # 青みがかった色
        point_color = (color_intensity, color_intensity, int(color_intensity * 0.5))

        points = {}
        for i, (x, y, z, v) in enumerate(lm):
            if v > 0.5:
                points[i] = (int(x * w), int(y * h))

        # 接続線
        for s, e in connections:
            if s in points and e in points:
                cv2.line(overlay, points[s], points[e], line_color, 1, cv2.LINE_AA)

        # 関節点
        for idx, (px, py) in points.items():
            cv2.circle(overlay, (px, py), 2, point_color, -1, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    return result


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


def _interpolate_tips(tips, max_gap=8):
    """検出が途切れたフレーム間を線形補間で埋める

    Args:
        tips: [(frame_idx, (x, y)), ...] 検出済みの先端座標
        max_gap: 補間する最大フレーム間隔（これ以上離れていたら補間しない）

    Returns:
        interpolated: [(frame_idx, (x, y)), ...] 補間済みリスト
    """
    if len(tips) < 2:
        return tips

    interpolated = [tips[0]]
    for j in range(1, len(tips)):
        prev_f, (px, py) = tips[j - 1]
        curr_f, (cx, cy) = tips[j]
        gap = curr_f - prev_f

        if 1 < gap <= max_gap:
            # 線形補間で中間フレームを生成
            for k in range(1, gap):
                t = k / gap
                ix = int(px + (cx - px) * t)
                iy = int(py + (cy - py) * t)
                interpolated.append((prev_f + k, (ix, iy)))

        interpolated.append(tips[j])

    return interpolated


def draw_detected_bat_path(frame, bat_detections, current_frame, trail_length=30):
    """YOLO検出結果に基づくバット先端軌道を描画（補間付き）

    Args:
        frame: BGR画像
        bat_detections: {frame_idx: detection_result} BatDetector の検出結果
        current_frame: 現在のフレーム番号
        trail_length: 軌跡の長さ（フレーム数）

    Returns:
        描画済みフレーム
    """
    overlay = frame.copy()
    trail_start = max(0, current_frame - trail_length)

    # 軌跡用の先端座標を収集
    raw_tips = []
    for f in range(trail_start, current_frame + 1):
        det = bat_detections.get(f)
        if det and det["tip"]:
            raw_tips.append((f, (int(det["tip"][0]), int(det["tip"][1]))))

    # 欠落フレームを線形補間で埋める
    tips = _interpolate_tips(raw_tips, max_gap=8)

    # 軌跡を緑系グラデーションで描画
    for j in range(1, len(tips)):
        _, pt = tips[j]
        _, prev_pt = tips[j - 1]
        alpha = (j + 1) / len(tips)
        color = (int(50 * alpha), int(255 * alpha), int(50 * alpha))
        thickness = max(1, int(4 * alpha))
        cv2.line(overlay, prev_pt, pt, color, thickness, cv2.LINE_AA)

    # 現在フレームの検出情報を描画
    det = bat_detections.get(current_frame)
    if det:
        tip = (int(det["tip"][0]), int(det["tip"][1]))
        handle = (int(det["handle"][0]), int(det["handle"][1]))

        # バットの線（グリップ→先端）
        cv2.line(overlay, handle, tip, (100, 255, 100), 2, cv2.LINE_AA)

        # 先端マーカー
        cv2.circle(overlay, tip, 8, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, tip, 8, (255, 255, 255), 2, cv2.LINE_AA)

        # バウンディングボックス（薄く）
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                      (100, 255, 100), 1, cv2.LINE_AA)

        # 信頼度ラベル
        conf = det["confidence"]
        cv2.putText(overlay, f"bat {conf:.0%}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    return result
