"""投球フォームのフェーズ検出・リリースポイント推定

投球を以下の5フェーズに分割:
1. ワインドアップ (Wind-up)
2. ストライド (Stride / Leg lift)
3. アームコッキング (Arm Cocking)
4. アーム加速 (Arm Acceleration)
5. フォロースルー (Follow-through)
"""

import numpy as np
from .angle_analyzer import calc_angle


# フェーズ定義
PITCHING_PHASES = {
    "windup":       {"name": "ワインドアップ",     "color": "#2196F3", "emoji": "🔄"},
    "leg_lift":     {"name": "足上げ",            "color": "#9C27B0", "emoji": "🦵"},
    "stride":       {"name": "ストライド",         "color": "#FF9800", "emoji": "🦶"},
    "arm_cocking":  {"name": "アームコッキング",    "color": "#E91E63", "emoji": "💪"},
    "acceleration": {"name": "腕加速",            "color": "#F44336", "emoji": "⚡"},
    "follow_through": {"name": "フォロースルー",   "color": "#4CAF50", "emoji": "🔄"},
}


def calc_throwing_arm_speed(landmarks_history, fps, arm="right"):
    """投げ腕の手首速度を計算

    Args:
        landmarks_history: 全フレームのlandmarks
        fps: FPS
        arm: "right" or "left"

    Returns:
        speeds: [(frame, speed), ...]
    """
    wrist_idx = 16 if arm == "right" else 15
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


def detect_pitch_motion(arm_speeds, fps, speed_threshold=0.6, min_frames=4):
    """投球動作の区間を検出

    Args:
        arm_speeds: calc_throwing_arm_speed() の戻り値
        fps: FPS
        speed_threshold: 速度閾値
        min_frames: 最小フレーム数

    Returns:
        pitches: [(start, end, release_frame, peak_speed), ...]
    """
    if not arm_speeds:
        return []

    all_speeds = [s for _, s in arm_speeds]
    mean_speed = np.mean(all_speeds)
    std_speed = np.std(all_speeds)
    adaptive = max(speed_threshold, mean_speed + 1.2 * std_speed)

    in_pitch = False
    pitches = []
    current_start = 0
    peak_frame = 0
    peak_speed = 0

    for frame, speed in arm_speeds:
        if speed > adaptive:
            if not in_pitch:
                current_start = frame
                peak_frame = frame
                peak_speed = speed
                in_pitch = True
            elif speed > peak_speed:
                peak_frame = frame
                peak_speed = speed
        else:
            if in_pitch:
                duration = frame - current_start
                if duration >= min_frames:
                    pitches.append((current_start, frame, peak_frame, peak_speed))
                in_pitch = False

    if in_pitch:
        last_frame = arm_speeds[-1][0]
        if last_frame - current_start >= min_frames:
            pitches.append((current_start, last_frame, peak_frame, peak_speed))

    return pitches


def detect_pitching_phases(landmarks_history, arm_speeds, pitch, fps, arm="right"):
    """投球をフェーズに分割

    Args:
        landmarks_history: 全フレームのlandmarks
        arm_speeds: calc_throwing_arm_speed() の戻り値
        pitch: (start, end, release, peak_speed)
        fps: FPS
        arm: "right" or "left"

    Returns:
        phases: [(phase_key, start_frame, end_frame), ...]
    """
    start, end, release, peak_speed = pitch
    speed_map = {f: s for f, s in arm_speeds}

    # インデックス設定
    if arm == "right":
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16
        hip_idx, knee_idx, ankle_idx = 24, 26, 28
        lead_knee_idx, lead_ankle_idx = 25, 27
    else:
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
        hip_idx, knee_idx, ankle_idx = 23, 25, 27
        lead_knee_idx, lead_ankle_idx = 26, 28

    pre_margin = int(fps * 0.8) if fps > 0 else 24
    post_margin = int(fps * 0.4) if fps > 0 else 12
    analysis_start = max(0, start - pre_margin)
    analysis_end = end + post_margin

    # === 足上げ検出: 踏み出し足が上がるフレーム ===
    leg_lift_start = analysis_start
    leg_lift_peak = analysis_start
    max_lift = 0

    for f in range(analysis_start, start):
        lm = landmarks_history.get(f)
        if lm is None:
            continue
        lead_knee = lm[lead_knee_idx]
        lead_ankle = lm[lead_ankle_idx]
        hip = lm[hip_idx]
        if lead_knee[3] > 0.5 and hip[3] > 0.5:
            # 膝が腰に近い = 足が上がっている（Y値が小さい = 上）
            lift = hip[1] - lead_knee[1]
            if lift > max_lift:
                max_lift = lift
                leg_lift_peak = f

    # 足上げ開始: 膝が上がり始めるフレーム
    for f in range(analysis_start, leg_lift_peak):
        lm = landmarks_history.get(f)
        next_lm = landmarks_history.get(f + 1)
        if lm and next_lm:
            if lm[lead_knee_idx][3] > 0.5 and next_lm[lead_knee_idx][3] > 0.5:
                dy = next_lm[lead_knee_idx][1] - lm[lead_knee_idx][1]
                if dy < -0.005:  # 膝が上がり始めた
                    leg_lift_start = f
                    break

    # === ストライド: 足が降りて地面に着くまで ===
    stride_start = leg_lift_peak + 1
    stride_end = start

    # 踏み出し足が下がっていく区間
    for f in range(leg_lift_peak + 1, start):
        lm = landmarks_history.get(f)
        if lm and lm[lead_ankle_idx][3] > 0.5:
            # 足首が元の高さに近づいたら着地
            ankle_y = lm[lead_ankle_idx][1]
            if ankle_y > 0.75:  # 画面下部 = 地面近く
                stride_end = f
                break

    # === アームコッキング: ストライド後〜腕が最大外旋するまで ===
    cocking_start = stride_end + 1
    cocking_end = start

    # 手首がY軸方向で最も高い位置（コッキング完了）
    max_height_frame = cocking_start
    min_wrist_y = 1.0
    for f in range(cocking_start, release):
        lm = landmarks_history.get(f)
        if lm and lm[wrist_idx][3] > 0.5:
            if lm[wrist_idx][1] < min_wrist_y:
                min_wrist_y = lm[wrist_idx][1]
                max_height_frame = f
    cocking_end = max_height_frame

    # === 腕加速: コッキング完了〜リリース ===
    accel_start = cocking_end + 1
    accel_end = release

    # === フォロースルー: リリース後 ===
    follow_start = release + 1

    # === フェーズリスト構築 ===
    phases = []

    # ワインドアップ
    if leg_lift_start > analysis_start:
        phases.append(("windup", analysis_start, leg_lift_start - 1))

    # 足上げ
    if stride_start > leg_lift_start:
        phases.append(("leg_lift", leg_lift_start, stride_start - 1))

    # ストライド
    if cocking_start > stride_start:
        phases.append(("stride", stride_start, cocking_start - 1))

    # アームコッキング
    if accel_start > cocking_start:
        phases.append(("arm_cocking", cocking_start, accel_start - 1))

    # 腕加速
    if accel_end >= accel_start:
        phases.append(("acceleration", accel_start, accel_end))

    # フォロースルー
    if analysis_end > follow_start:
        phases.append(("follow_through", follow_start, analysis_end))

    return phases


def get_pitching_phase_at_frame(phases, frame_idx):
    """指定フレームのフェーズを取得"""
    for key, start, end in phases:
        if start <= frame_idx <= end:
            return key, PITCHING_PHASES[key]
    return None, None


def detect_release_point(landmarks_history, pitch, fps, arm="right"):
    """リリースポイントを詳細に推定

    手首の最大速度フレーム周辺で、手首が頭の前方にある位置。

    Args:
        landmarks_history: 全フレームのlandmarks
        pitch: (start, end, release_frame, peak_speed)
        fps: FPS
        arm: "right" or "left"

    Returns:
        release_info: {
            "frame": int,
            "position": (x, y),
            "height_ratio": float,  # 身長に対する高さ比率
            "forward_distance": float,  # 頭からの前方距離
            "elbow_angle": float,
            "shoulder_angle": float,
        }
    """
    _, _, release_frame, _ = pitch
    wrist_idx = 16 if arm == "right" else 15
    shoulder_idx = 12 if arm == "right" else 11
    elbow_idx = 14 if arm == "right" else 13
    hip_idx = 24 if arm == "right" else 23

    lm = landmarks_history.get(release_frame)
    if lm is None:
        return None

    wrist = lm[wrist_idx]
    shoulder = lm[shoulder_idx]
    elbow = lm[elbow_idx]
    hip = lm[hip_idx]
    nose = lm[0]

    if any(pt[3] < 0.5 for pt in (wrist, shoulder, elbow, nose)):
        return None

    # 肘角度
    elbow_angle = calc_angle(
        (shoulder[0], shoulder[1]),
        (elbow[0], elbow[1]),
        (wrist[0], wrist[1]),
    )

    # 肩角度
    shoulder_angle = calc_angle(
        (elbow[0], elbow[1]),
        (shoulder[0], shoulder[1]),
        (hip[0], hip[1]),
    ) if hip[3] > 0.5 else None

    # 身長推定: 頭〜足首
    ankle = lm[28 if arm == "right" else 27]
    if nose[3] > 0.5 and ankle[3] > 0.5:
        body_height = abs(ankle[1] - nose[1])
        if body_height > 0.01:
            # リリース高さ = 身長に対する比率
            release_height = abs(ankle[1] - wrist[1])
            height_ratio = release_height / body_height
        else:
            height_ratio = None
    else:
        height_ratio = None

    # 頭からの前方距離
    forward_dist = wrist[0] - nose[0]

    return {
        "frame": release_frame,
        "position": (wrist[0], wrist[1]),
        "height_ratio": height_ratio,
        "forward_distance": forward_dist,
        "elbow_angle": elbow_angle,
        "shoulder_angle": shoulder_angle,
    }


def calc_arm_slot(landmarks_history, release_frame, arm="right"):
    """アームスロット（腕の角度）を推定

    リリース時の腕の角度。
    0° = サイドスロー、90° = オーバースロー

    Returns:
        arm_slot: 角度（度）、None if 検出不可
    """
    lm = landmarks_history.get(release_frame)
    if lm is None:
        return None

    shoulder_idx = 12 if arm == "right" else 11
    wrist_idx = 16 if arm == "right" else 15

    shoulder = lm[shoulder_idx]
    wrist = lm[wrist_idx]

    if shoulder[3] < 0.5 or wrist[3] < 0.5:
        return None

    dx = wrist[0] - shoulder[0]
    dy = shoulder[1] - wrist[1]  # Y反転（上が正）

    angle = np.degrees(np.arctan2(dy, abs(dx)))
    return angle
