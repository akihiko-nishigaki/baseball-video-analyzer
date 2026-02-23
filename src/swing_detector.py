"""バッティングスイングの自動検出・区間分割"""

import numpy as np


def calc_wrist_speed(landmarks_history, fps, wrist_idx=16):
    """手首の速度推移を計算

    Args:
        landmarks_history: {frame_idx: landmarks} のdict
        fps: 動画のFPS
        wrist_idx: 手首の関節インデックス（16=右手首, 15=左手首）

    Returns:
        speeds: [(frame_idx, speed), ...] のリスト
    """
    frames = sorted(landmarks_history.keys())
    speeds = []

    for i in range(1, len(frames)):
        prev_f = frames[i - 1]
        curr_f = frames[i]

        prev_lm = landmarks_history[prev_f]
        curr_lm = landmarks_history[curr_f]

        if prev_lm is None or curr_lm is None:
            speeds.append((curr_f, 0.0))
            continue

        if prev_lm[wrist_idx][3] < 0.5 or curr_lm[wrist_idx][3] < 0.5:
            speeds.append((curr_f, 0.0))
            continue

        dx = curr_lm[wrist_idx][0] - prev_lm[wrist_idx][0]
        dy = curr_lm[wrist_idx][1] - prev_lm[wrist_idx][1]
        dt = (curr_f - prev_f) / fps if fps > 0 else 1

        speed = np.sqrt(dx**2 + dy**2) / dt
        speeds.append((curr_f, speed))

    return speeds


def detect_swings(wrist_speeds, fps, speed_threshold=0.8, min_swing_frames=5):
    """スイング区間を自動検出

    手首の速度が閾値を超えている連続区間をスイングとして検出。

    Args:
        wrist_speeds: calc_wrist_speed() の戻り値
        fps: 動画のFPS
        speed_threshold: スイング判定の速度閾値（正規化座標/秒）
        min_swing_frames: 最小スイングフレーム数

    Returns:
        swings: [(start_frame, end_frame, peak_frame, peak_speed), ...]
    """
    if not wrist_speeds:
        return []

    # 速度の平均・標準偏差で閾値を自動調整
    all_speeds = [s for _, s in wrist_speeds]
    mean_speed = np.mean(all_speeds)
    std_speed = np.std(all_speeds)
    adaptive_threshold = max(speed_threshold, mean_speed + 1.5 * std_speed)

    # 閾値を超える区間を検出
    in_swing = False
    swings = []
    current_start = 0
    current_peak = 0
    current_peak_speed = 0

    for frame, speed in wrist_speeds:
        if speed > adaptive_threshold:
            if not in_swing:
                current_start = frame
                current_peak = frame
                current_peak_speed = speed
                in_swing = True
            elif speed > current_peak_speed:
                current_peak = frame
                current_peak_speed = speed
        else:
            if in_swing:
                duration = frame - current_start
                if duration >= min_swing_frames:
                    swings.append((current_start, frame, current_peak, current_peak_speed))
                in_swing = False

    # 最後のスイング
    if in_swing:
        last_frame = wrist_speeds[-1][0]
        duration = last_frame - current_start
        if duration >= min_swing_frames:
            swings.append((current_start, last_frame, current_peak, current_peak_speed))

    return swings


def calc_swing_metrics(landmarks_history, swing, fps):
    """スイングの詳細メトリクスを計算

    Args:
        landmarks_history: 全フレームの関節データ
        swing: (start, end, peak, peak_speed) タプル
        fps: FPS

    Returns:
        metrics dict
    """
    start, end, peak, peak_speed = swing

    metrics = {
        "開始フレーム": start,
        "終了フレーム": end,
        "インパクト推定": peak,
        "スイング時間": f"{(end - start) / fps:.2f}秒" if fps > 0 else "N/A",
        "最高速度": f"{peak_speed:.2f}",
    }

    # インパクト時の姿勢チェック
    peak_lm = landmarks_history.get(peak)
    if peak_lm:
        from .angle_analyzer import calc_angle

        # 前肘（左打者=右肘、右打者=左肘 — ここでは両方表示）
        for side, (s_idx, e_idx, w_idx) in [("右肘", (12, 14, 16)), ("左肘", (11, 13, 15))]:
            if all(peak_lm[i][3] > 0.5 for i in (s_idx, e_idx, w_idx)):
                angle = calc_angle(
                    (peak_lm[s_idx][0], peak_lm[s_idx][1]),
                    (peak_lm[e_idx][0], peak_lm[e_idx][1]),
                    (peak_lm[w_idx][0], peak_lm[w_idx][1]),
                )
                metrics[f"インパクト時{side}角度"] = f"{angle:.1f}°"

        # ステップ幅
        la = peak_lm[27]  # 左足首
        ra = peak_lm[28]  # 右足首
        if la[3] > 0.5 and ra[3] > 0.5:
            step_width = abs(la[0] - ra[0])
            # 肩幅との比率
            ls = peak_lm[11]
            rs = peak_lm[12]
            if ls[3] > 0.5 and rs[3] > 0.5:
                shoulder_width = abs(ls[0] - rs[0])
                if shoulder_width > 0:
                    ratio = step_width / shoulder_width
                    metrics["ステップ幅/肩幅"] = f"{ratio:.2f}倍"

        # 体の開き
        ls = peak_lm[11]
        rs = peak_lm[12]
        if ls[3] > 0.5 and rs[3] > 0.5:
            dx = rs[0] - ls[0]
            dy = rs[1] - ls[1]
            rotation = np.degrees(np.arctan2(abs(dy), abs(dx)))
            metrics["インパクト時肩回旋"] = f"{rotation:.1f}°"

    return metrics


def calc_weight_shift(landmarks_history, swing):
    """スイング中の体重移動を分析

    Args:
        landmarks_history: 全フレームの関節データ
        swing: (start, end, peak, peak_speed) タプル

    Returns:
        weight_data: [(frame, cog_x, rear_weight_ratio), ...]
    """
    start, end, _, _ = swing
    weight_data = []

    for f in range(start, end + 1):
        lm = landmarks_history.get(f)
        if lm is None:
            continue

        lh = lm[23]  # 左腰
        rh = lm[24]  # 右腰
        la = lm[27]  # 左足首
        ra = lm[28]  # 右足首

        if any(pt[3] < 0.5 for pt in (lh, rh, la, ra)):
            continue

        # 重心X（両腰の中点）
        cog_x = (lh[0] + rh[0]) / 2
        # 両足の中点
        foot_center = (la[0] + ra[0]) / 2
        # 後ろ足方向にどれだけ重心があるか
        foot_span = abs(la[0] - ra[0])
        if foot_span > 0.01:
            # 0.5 = 中央、0 = 前足寄り、1 = 後ろ足寄り
            shift_ratio = (cog_x - min(la[0], ra[0])) / foot_span
        else:
            shift_ratio = 0.5

        weight_data.append((f, cog_x, shift_ratio))

    return weight_data
