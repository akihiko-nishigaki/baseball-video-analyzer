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


def _create_kalman_filter():
    """バット先端追跡用2D Kalmanフィルター (位置+速度)"""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def _get_wrist_info(lm, frame_shape):
    """手首と体の基準点をピクセル座標で取得

    Returns:
        dict with 'wrist' and 'body_center' keys, or None
    """
    VIS = 0.2
    h, w = frame_shape[:2]
    rw = lm[16]
    if rw[3] < VIS:
        return None

    wrist_x = int(rw[0] * w)
    wrist_y = int(rw[1] * h)

    body_cx, body_cy = None, None
    if lm[23][3] > VIS and lm[24][3] > VIS:
        body_cx = int((lm[23][0] + lm[24][0]) / 2 * w)
        body_cy = int((lm[23][1] + lm[24][1]) / 2 * h)
    elif lm[11][3] > VIS and lm[12][3] > VIS:
        body_cx = int((lm[11][0] + lm[12][0]) / 2 * w)
        body_cy = int((lm[11][1] + lm[12][1]) / 2 * h)

    return {
        'wrist': (wrist_x, wrist_y),
        'body_center': (body_cx, body_cy) if body_cx is not None else None
    }


def _detect_tip_ellipse(motion_mask, lm, frame_shape):
    """Stage 2: fitEllipseで棒状輪郭からバット先端を検出

    モーションマスクの輪郭に楕円をフィットし、
    アスペクト比 > 2.5 の棒状領域の長軸端点のうち
    手首から遠い方をバット先端とする。
    """
    info = _get_wrist_info(lm, frame_shape)
    if info is None:
        return None

    wrist_x, wrist_y = info['wrist']
    h, w = frame_shape[:2]

    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_tip = None
    best_score = 0

    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < 100 or area > w * h * 0.3:
            continue

        ellipse = cv2.fitEllipse(cnt)
        center, (d1, d2), angle = ellipse

        if min(d1, d2) < 1:
            continue
        aspect = max(d1, d2) / min(d1, d2)

        if aspect < 2.5:
            continue

        # 長軸の2端点を算出
        if d1 >= d2:
            half = d1 / 2
            a_rad = np.radians(angle)
        else:
            half = d2 / 2
            a_rad = np.radians(angle + 90)

        dx = half * np.cos(a_rad)
        dy = half * np.sin(a_rad)
        ep1 = (int(center[0] + dx), int(center[1] + dy))
        ep2 = (int(center[0] - dx), int(center[1] - dy))

        # 手首から遠い端点 = バット先端
        dist1 = np.sqrt((ep1[0] - wrist_x)**2 + (ep1[1] - wrist_y)**2)
        dist2 = np.sqrt((ep2[0] - wrist_x)**2 + (ep2[1] - wrist_y)**2)

        if dist1 > dist2:
            tip, tip_dist = ep1, dist1
        else:
            tip, tip_dist = ep2, dist2

        if tip_dist < 30:
            continue
        if not (0 <= tip[0] < w and 0 <= tip[1] < h):
            continue

        score = aspect * tip_dist
        if score > best_score:
            best_score = score
            best_tip = tip

    return best_tip


def _detect_tip_lsd(gray, motion_mask, lm, frame_shape):
    """Stage 3: LSD線分検出でモーションブラーの線からバット先端を検出

    動き領域内で検出した線分のうち、一端が手首付近にあり
    最も長い線分の遠端をバット先端とする。
    """
    info = _get_wrist_info(lm, frame_shape)
    if info is None:
        return None

    wrist_x, wrist_y = info['wrist']
    h, w = frame_shape[:2]

    masked = cv2.bitwise_and(gray, gray, mask=motion_mask)

    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(masked)[0]
    except Exception:
        return None

    if lines is None:
        return None

    best_tip = None
    best_score = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length < 30:
            continue

        d1w = np.sqrt((x1 - wrist_x)**2 + (y1 - wrist_y)**2)
        d2w = np.sqrt((x2 - wrist_x)**2 + (y2 - wrist_y)**2)

        # 少なくとも一端が手首からフレーム幅30%以内
        if min(d1w, d2w) > w * 0.3:
            continue

        # 手首から遠い端 = 先端
        if d1w > d2w:
            tip = (int(x1), int(y1))
            tip_dist = d1w
        else:
            tip = (int(x2), int(y2))
            tip_dist = d2w

        if not (0 <= tip[0] < w and 0 <= tip[1] < h):
            continue

        score = length * tip_dist
        if score > best_score:
            best_score = score
            best_tip = tip

    return best_tip


def _detect_tip_optical_flow(prev_gray, curr_gray, lm, frame_shape):
    """Stage 0: Dense Optical Flowの最大速度点をバット先端とする

    バット先端は回転運動の最外点であり、物理的に最も速く動く。
    Farneback dense optical flowで各ピクセルの速度を計算し、
    手首外側方向で「速度×距離」スコアが最大の領域をバット先端とする。
    """
    info = _get_wrist_info(lm, frame_shape)
    if info is None:
        return None

    wrist_x, wrist_y = info['wrist']
    h, w = frame_shape[:2]

    # Dense optical flow (大きな変位に対応するパラメータ)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    # フロー速度（magnitude）
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # ノイズ除去: 最低速度閾値
    motion_mask = mag > 3.0
    ys, xs = np.where(motion_mask)
    if len(ys) < 10:
        return None

    # 体→手首方向でフィルタ（手首より外側の点のみ）
    body_center = info.get('body_center')
    if body_center is not None:
        body_cx, body_cy = body_center
        out_dx = float(wrist_x - body_cx)
        out_dy = float(wrist_y - body_cy)
        out_len = np.sqrt(out_dx**2 + out_dy**2)
        if out_len > 10:
            out_dx /= out_len
            out_dy /= out_len
            rel_x = xs.astype(np.float32) - wrist_x
            rel_y = ys.astype(np.float32) - wrist_y
            dots = rel_x * out_dx + rel_y * out_dy
            outward = dots > 0
            if np.any(outward):
                xs = xs[outward]
                ys = ys[outward]

    if len(ys) < 5:
        return None

    mags = mag[ys, xs]
    dists = np.sqrt((xs.astype(np.float32) - wrist_x)**2 +
                    (ys.astype(np.float32) - wrist_y)**2)

    # 手自体を除外
    far_enough = dists > 30
    if not np.any(far_enough):
        return None

    xs = xs[far_enough]
    ys = ys[far_enough]
    mags = mags[far_enough]
    dists = dists[far_enough]

    # スコア = 速度 × 距離（バット先端 = 速い + 遠い）
    scores = mags * dists

    # 上位3%の重心
    threshold = np.percentile(scores, 97)
    top = scores >= threshold
    if not np.any(top):
        return None

    tip_x = int(np.mean(xs[top]))
    tip_y = int(np.mean(ys[top]))

    if not (0 <= tip_x < w and 0 <= tip_y < h):
        return None

    return (tip_x, tip_y)


def _detect_tip_farthest(motion_mask, lm, frame_shape):
    """Stage 4: 手首外側方向の最遠動きピクセルをバット先端とする

    体の中心→手首方向に延長し、その方向の動きピクセルの
    上位5%の重心をバット先端とする（最終フォールバック）。
    """
    info = _get_wrist_info(lm, frame_shape)
    if info is None or info['body_center'] is None:
        return None

    wrist_x, wrist_y = info['wrist']
    body_cx, body_cy = info['body_center']
    h, w = frame_shape[:2]

    ys, xs = np.where(motion_mask > 0)
    if len(ys) < 30:
        return None

    out_dx = float(wrist_x - body_cx)
    out_dy = float(wrist_y - body_cy)
    out_len = np.sqrt(out_dx**2 + out_dy**2)
    if out_len < 10:
        return None
    out_dx /= out_len
    out_dy /= out_len

    rel_x = xs.astype(np.float32) - wrist_x
    rel_y = ys.astype(np.float32) - wrist_y

    dots = rel_x * out_dx + rel_y * out_dy
    outward = dots > 0

    if not np.any(outward):
        return None

    out_xs = xs[outward]
    out_ys = ys[outward]
    dists = np.sqrt((out_xs - wrist_x)**2 + (out_ys - wrist_y)**2)

    min_dist = max(30, out_len * 0.3)
    far_enough = dists > min_dist

    if not np.any(far_enough):
        return None

    far_dists = dists[far_enough]
    far_xs = out_xs[far_enough]
    far_ys = out_ys[far_enough]

    threshold = np.percentile(far_dists, 95)
    top = far_dists >= threshold

    tip_x = int(np.mean(far_xs[top]))
    tip_y = int(np.mean(far_ys[top]))

    if not (0 <= tip_x < w and 0 <= tip_y < h):
        return None

    return (tip_x, tip_y)


def compute_motion_bat_tips(reader, landmarks_history, progress_cb=None):
    """多段パイプラインでバット先端位置を検出

    Pipeline:
    1. 3-frame differencing + CLAHE → motion mask
    2. Contour + fitEllipse → 棒状領域の長軸端点
    3. LSD line detection → ブラーストリークの線端点
    4. Farthest outward motion → 最遠動きピクセル重心
    + Kalman filter で軌道予測・平滑化

    Args:
        reader: VideoReader
        landmarks_history: {frame_idx: landmarks}
        progress_cb: callback(current, total)

    Returns:
        {frame_idx: (tip_x, tip_y)} ピクセル座標
    """
    tips = {}
    gray_buf = [None, None, None]
    total = reader.total_frames

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kf = _create_kalman_filter()
    kf_init = False
    miss = 0
    MAX_MISS = 8

    for frame_idx, frame in reader.iter_frames():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3フレームバッファをシフト
        gray_buf[0] = gray_buf[1]
        gray_buf[1] = gray_buf[2]
        gray_buf[2] = gray

        if gray_buf[0] is None:
            if progress_cb and frame_idx % 10 == 0:
                progress_cb(frame_idx, total)
            continue

        lm = landmarks_history.get(frame_idx)
        if lm is None:
            miss += 1
            if progress_cb and frame_idx % 10 == 0:
                progress_cb(frame_idx, total)
            continue

        # 3-frame differencing (cavity問題を解決)
        d1 = cv2.absdiff(gray_buf[0], gray_buf[1])
        d2 = cv2.absdiff(gray_buf[1], gray_buf[2])
        motion = cv2.bitwise_and(d1, d2)
        _, motion = cv2.threshold(motion, 15, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
        motion = cv2.dilate(motion, kernel, iterations=2)

        # 多段検出: OpticalFlow → Ellipse → LSD → Farthest
        tip = _detect_tip_optical_flow(gray_buf[1], gray_buf[2], lm, frame.shape)
        if tip is None:
            tip = _detect_tip_ellipse(motion, lm, frame.shape)
        if tip is None:
            tip = _detect_tip_lsd(gray_buf[2], motion, lm, frame.shape)
        if tip is None:
            tip = _detect_tip_farthest(motion, lm, frame.shape)

        # Kalmanフィルター更新
        if tip is not None:
            if not kf_init:
                kf.statePost = np.array(
                    [[np.float32(tip[0])], [np.float32(tip[1])],
                     [0.], [0.]], dtype=np.float32)
                kf_init = True
            else:
                kf.predict()
                kf.correct(np.array(
                    [[np.float32(tip[0])], [np.float32(tip[1])]],
                    dtype=np.float32))
            tips[frame_idx] = tip
            miss = 0
        elif kf_init and miss < MAX_MISS:
            pred = kf.predict()
            px, py = int(pred[0, 0]), int(pred[1, 0])
            fh, fw = frame.shape[:2]
            if 0 <= px < fw and 0 <= py < fh:
                tips[frame_idx] = (px, py)
            miss += 1
        else:
            miss += 1

        if progress_cb and frame_idx % 10 == 0:
            progress_cb(frame_idx, total)

    return tips


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
