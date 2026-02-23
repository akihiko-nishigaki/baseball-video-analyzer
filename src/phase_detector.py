"""バッティングフェーズの自動検出

スイングを以下の5フェーズに分割:
1. 構え (Stance)
2. テイクバック (Load/Takeback)
3. ステップ (Stride)
4. インパクト (Swing/Impact)
5. フォロースルー (Follow-through)
"""

import numpy as np


# フェーズ定義
BATTING_PHASES = {
    "stance":        {"name": "構え",          "color": "#2196F3", "emoji": "🧍"},
    "load":          {"name": "テイクバック",    "color": "#FF9800", "emoji": "↩️"},
    "stride":        {"name": "ステップ",       "color": "#9C27B0", "emoji": "🦶"},
    "swing":         {"name": "スイング",       "color": "#F44336", "emoji": "💥"},
    "follow_through": {"name": "フォロースルー", "color": "#4CAF50", "emoji": "🔄"},
}


def detect_batting_phases(landmarks_history, wrist_speeds, swing, fps):
    """スイング区間をフェーズに分割

    Args:
        landmarks_history: 全フレームの関節データ
        wrist_speeds: calc_wrist_speed() の戻り値
        swing: (start, end, peak, peak_speed) タプル
        fps: FPS

    Returns:
        phases: [(phase_key, start_frame, end_frame), ...]
    """
    start, end, peak, peak_speed = swing

    # スイング前後の余白を含める
    pre_margin = int(fps * 0.5) if fps > 0 else 15  # 0.5秒前
    post_margin = int(fps * 0.3) if fps > 0 else 10  # 0.3秒後
    analysis_start = max(0, start - pre_margin)
    analysis_end = end + post_margin

    # 速度データをフレーム番号でルックアップ可能に
    speed_map = {f: s for f, s in wrist_speeds}

    # === フェーズ境界の推定 ===

    # 1. 手首の後方移動（テイクバック）を検出
    #    → 手首がスイング方向と逆に動く区間
    load_start = analysis_start
    load_end = start

    # テイクバック開始: 手首が後ろに動き始めるフレーム
    for f in range(analysis_start, start):
        lm = landmarks_history.get(f)
        next_lm = landmarks_history.get(f + 1)
        if lm and next_lm and lm[16][3] > 0.5 and next_lm[16][3] > 0.5:
            dx = next_lm[16][0] - lm[16][0]
            # 手首が後方に動き始めた = テイクバック開始
            # (方向は打者の向きに依存するが、速度変化で判定)
            if abs(dx) > 0.005:
                load_start = f
                break

    # 2. ステップ開始: 前足が持ち上がる or 前方に動き始める
    stride_start = start - int(fps * 0.15) if fps > 0 else start - 5
    stride_start = max(analysis_start, stride_start)

    # 前足の動きを検出
    for f in range(load_end, start):
        lm = landmarks_history.get(f)
        next_lm = landmarks_history.get(f + 1)
        if lm and next_lm:
            # 左足首(27)または右足首(28)の上下動
            for ankle_idx in (27, 28):
                if lm[ankle_idx][3] > 0.5 and next_lm[ankle_idx][3] > 0.5:
                    dy = next_lm[ankle_idx][1] - lm[ankle_idx][1]
                    if dy < -0.01:  # 足が上がった
                        stride_start = f
                        break

    # 3. スイング開始: 手首速度が急上昇
    swing_start = start
    speed_threshold = peak_speed * 0.3
    for f in range(stride_start, peak):
        s = speed_map.get(f, 0)
        if s > speed_threshold:
            swing_start = f
            break

    # 4. インパクト: 速度ピーク付近
    impact_frame = peak

    # 5. フォロースルー: インパクト後
    follow_start = peak + 1

    # === フェーズリスト構築 ===
    phases = []

    # 構え: analysis_start → load_start
    if load_start > analysis_start:
        phases.append(("stance", analysis_start, load_start - 1))

    # テイクバック: load_start → stride_start
    if stride_start > load_start:
        phases.append(("load", load_start, stride_start - 1))

    # ステップ: stride_start → swing_start
    if swing_start > stride_start:
        phases.append(("stride", stride_start, swing_start - 1))

    # スイング: swing_start → impact
    phases.append(("swing", swing_start, impact_frame))

    # フォロースルー: impact → analysis_end
    if analysis_end > follow_start:
        phases.append(("follow_through", follow_start, analysis_end))

    return phases


def get_phase_at_frame(phases, frame_idx):
    """指定フレームのフェーズを取得

    Returns:
        (phase_key, phase_info) or (None, None)
    """
    for key, start, end in phases:
        if start <= frame_idx <= end:
            return key, BATTING_PHASES[key]
    return None, None


def get_phase_checkpoints(landmarks_history, phases):
    """各フェーズの代表フレームでのチェックポイント

    Returns:
        checkpoints: [{phase, frame, checks: [{item, value, status, advice}]}, ...]
    """
    from .angle_analyzer import calc_angle

    checkpoints = []

    for phase_key, start, end in phases:
        mid = (start + end) // 2
        lm = landmarks_history.get(mid)
        if lm is None:
            continue

        checks = []

        if phase_key == "stance":
            # 構えチェック: 膝の曲がり、重心の低さ
            for side, (h, k, a) in [("右膝", (24, 26, 28)), ("左膝", (23, 25, 27))]:
                if all(lm[i][3] > 0.5 for i in (h, k, a)):
                    angle = calc_angle(
                        (lm[h][0], lm[h][1]),
                        (lm[k][0], lm[k][1]),
                        (lm[a][0], lm[a][1]),
                    )
                    if 130 <= angle <= 155:
                        status = "good"
                        advice = "適度に膝が曲がっています"
                    elif angle > 165:
                        status = "warning"
                        advice = "膝をもう少し曲げましょう"
                    else:
                        status = "info"
                        advice = "深く曲げすぎかもしれません"

                    checks.append({
                        "item": f"{side}の角度",
                        "value": f"{angle:.0f}°",
                        "status": status,
                        "advice": advice,
                    })

        elif phase_key == "load":
            # テイクバックチェック: 手首の位置、体の捻り
            ls = lm[11]
            rs = lm[12]
            if ls[3] > 0.5 and rs[3] > 0.5:
                dx = rs[0] - ls[0]
                dy = rs[1] - ls[1]
                rot = np.degrees(np.arctan2(abs(dy), abs(dx)))
                if rot < 20:
                    status = "good"
                    advice = "しっかり体を捻れています"
                else:
                    status = "warning"
                    advice = "もう少し体を捻ると力が溜まります"

                checks.append({
                    "item": "体の捻り",
                    "value": f"{rot:.0f}°",
                    "status": status,
                    "advice": advice,
                })

        elif phase_key == "swing":
            # インパクト付近: 肘の角度、体の開き
            for side, (s, e, w) in [("前肘", (11, 13, 15)), ("後ろ肘", (12, 14, 16))]:
                if all(lm[i][3] > 0.5 for i in (s, e, w)):
                    angle = calc_angle(
                        (lm[s][0], lm[s][1]),
                        (lm[e][0], lm[e][1]),
                        (lm[w][0], lm[w][1]),
                    )
                    if 140 <= angle <= 175:
                        status = "good"
                        advice = "肘がしっかり伸びています"
                    elif angle < 120:
                        status = "warning"
                        advice = "肘が曲がりすぎています"
                    else:
                        status = "info"
                        advice = ""

                    checks.append({
                        "item": f"{side}の角度",
                        "value": f"{angle:.0f}°",
                        "status": status,
                        "advice": advice,
                    })

        elif phase_key == "follow_through":
            # フォロースルー: 腕の伸び
            for s, e, w in [(12, 14, 16), (11, 13, 15)]:
                if all(lm[i][3] > 0.5 for i in (s, e, w)):
                    angle = calc_angle(
                        (lm[s][0], lm[s][1]),
                        (lm[e][0], lm[e][1]),
                        (lm[w][0], lm[w][1]),
                    )
                    if angle > 150:
                        checks.append({
                            "item": "腕の伸び",
                            "value": f"{angle:.0f}°",
                            "status": "good",
                            "advice": "最後までしっかり振り切れています",
                        })
                    break

        if checks:
            checkpoints.append({
                "phase": phase_key,
                "phase_name": BATTING_PHASES[phase_key]["name"],
                "frame": mid,
                "checks": checks,
            })

    return checkpoints
