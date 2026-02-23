"""バッティングフォームの総合評価"""

import numpy as np
from .angle_analyzer import calc_angle


# 評価基準（少年野球向け、厳しすぎない設定）
CRITERIA = {
    "knee_bend": {
        "name": "膝の使い方",
        "ideal_range": (130, 155),
        "max_score": 20,
        "advice_good": "膝が適度に曲がり、安定した構えです",
        "advice_warn": "膝の曲がりを意識しましょう",
    },
    "elbow_extension": {
        "name": "肘の伸び",
        "ideal_range": (140, 175),
        "max_score": 20,
        "advice_good": "インパクトで肘がしっかり伸びています",
        "advice_warn": "インパクトでもう少し肘を伸ばしましょう",
    },
    "step_width": {
        "name": "ステップ幅",
        "ideal_range": (1.0, 1.6),  # 肩幅比
        "max_score": 15,
        "advice_good": "適切なステップ幅です",
        "advice_warn": "ステップ幅を調整しましょう",
    },
    "weight_shift": {
        "name": "体重移動",
        "max_score": 20,
        "advice_good": "後ろ足から前足へスムーズに体重移動できています",
        "advice_warn": "体重移動を意識しましょう",
    },
    "follow_through": {
        "name": "振り切り",
        "max_score": 15,
        "advice_good": "最後まで振り切れています",
        "advice_warn": "最後までしっかり振り切りましょう",
    },
    "head_stability": {
        "name": "頭の安定",
        "max_score": 10,
        "advice_good": "頭がブレずに安定しています",
        "advice_warn": "スイング中に頭が動きすぎています",
    },
}


def evaluate_batting(landmarks_history, swing, weight_data=None):
    """バッティングフォームを総合評価

    Args:
        landmarks_history: 全フレームのlandmarks
        swing: (start, end, peak, peak_speed)
        weight_data: calc_weight_shift() の戻り値

    Returns:
        evaluation: {
            "total_score": int (0-100),
            "grade": str,
            "details": [{"name", "score", "max", "status", "advice"}, ...],
            "summary": str,
        }
    """
    start, end, peak, _ = swing
    details = []

    # === 1. 膝の使い方 ===
    knee_score = 0
    # 構え時の膝角度をチェック
    stance_frame = start - 5 if start > 5 else start
    lm = landmarks_history.get(stance_frame)
    if lm:
        for h, k, a in [(24, 26, 28), (23, 25, 27)]:
            if all(lm[i][3] > 0.5 for i in (h, k, a)):
                angle = calc_angle(
                    (lm[h][0], lm[h][1]),
                    (lm[k][0], lm[k][1]),
                    (lm[a][0], lm[a][1]),
                )
                low, high = CRITERIA["knee_bend"]["ideal_range"]
                if low <= angle <= high:
                    knee_score = CRITERIA["knee_bend"]["max_score"]
                elif abs(angle - low) <= 20 or abs(angle - high) <= 20:
                    knee_score = int(CRITERIA["knee_bend"]["max_score"] * 0.6)
                else:
                    knee_score = int(CRITERIA["knee_bend"]["max_score"] * 0.3)
                break

    details.append({
        "name": CRITERIA["knee_bend"]["name"],
        "score": knee_score,
        "max": CRITERIA["knee_bend"]["max_score"],
        "status": "good" if knee_score >= 15 else "warning" if knee_score >= 10 else "bad",
        "advice": CRITERIA["knee_bend"]["advice_good"] if knee_score >= 15
                  else CRITERIA["knee_bend"]["advice_warn"],
    })

    # === 2. 肘の伸び（インパクト時） ===
    elbow_score = 0
    peak_lm = landmarks_history.get(peak)
    if peak_lm:
        best_angle = 0
        for s, e, w in [(12, 14, 16), (11, 13, 15)]:
            if all(peak_lm[i][3] > 0.5 for i in (s, e, w)):
                angle = calc_angle(
                    (peak_lm[s][0], peak_lm[s][1]),
                    (peak_lm[e][0], peak_lm[e][1]),
                    (peak_lm[w][0], peak_lm[w][1]),
                )
                best_angle = max(best_angle, angle)

        low, high = CRITERIA["elbow_extension"]["ideal_range"]
        if low <= best_angle <= high:
            elbow_score = CRITERIA["elbow_extension"]["max_score"]
        elif abs(best_angle - low) <= 20:
            elbow_score = int(CRITERIA["elbow_extension"]["max_score"] * 0.6)
        else:
            elbow_score = int(CRITERIA["elbow_extension"]["max_score"] * 0.3)

    details.append({
        "name": CRITERIA["elbow_extension"]["name"],
        "score": elbow_score,
        "max": CRITERIA["elbow_extension"]["max_score"],
        "status": "good" if elbow_score >= 15 else "warning" if elbow_score >= 10 else "bad",
        "advice": CRITERIA["elbow_extension"]["advice_good"] if elbow_score >= 15
                  else CRITERIA["elbow_extension"]["advice_warn"],
    })

    # === 3. ステップ幅 ===
    step_score = 0
    if peak_lm:
        la = peak_lm[27]
        ra = peak_lm[28]
        ls = peak_lm[11]
        rs = peak_lm[12]
        if all(pt[3] > 0.5 for pt in (la, ra, ls, rs)):
            step_w = abs(la[0] - ra[0])
            shoulder_w = abs(ls[0] - rs[0])
            if shoulder_w > 0.01:
                ratio = step_w / shoulder_w
                low, high = CRITERIA["step_width"]["ideal_range"]
                if low <= ratio <= high:
                    step_score = CRITERIA["step_width"]["max_score"]
                elif abs(ratio - low) <= 0.3 or abs(ratio - high) <= 0.3:
                    step_score = int(CRITERIA["step_width"]["max_score"] * 0.6)
                else:
                    step_score = int(CRITERIA["step_width"]["max_score"] * 0.3)

    details.append({
        "name": CRITERIA["step_width"]["name"],
        "score": step_score,
        "max": CRITERIA["step_width"]["max_score"],
        "status": "good" if step_score >= 12 else "warning" if step_score >= 8 else "bad",
        "advice": CRITERIA["step_width"]["advice_good"] if step_score >= 12
                  else CRITERIA["step_width"]["advice_warn"],
    })

    # === 4. 体重移動 ===
    weight_score = 0
    if weight_data and len(weight_data) >= 3:
        # スイング開始時と終了時の重心位置を比較
        start_ratio = weight_data[0][2]
        end_ratio = weight_data[-1][2]
        shift = abs(end_ratio - start_ratio)

        if shift > 0.15:  # 十分な体重移動
            weight_score = CRITERIA["weight_shift"]["max_score"]
        elif shift > 0.08:
            weight_score = int(CRITERIA["weight_shift"]["max_score"] * 0.6)
        else:
            weight_score = int(CRITERIA["weight_shift"]["max_score"] * 0.3)

    details.append({
        "name": CRITERIA["weight_shift"]["name"],
        "score": weight_score,
        "max": CRITERIA["weight_shift"]["max_score"],
        "status": "good" if weight_score >= 15 else "warning" if weight_score >= 10 else "bad",
        "advice": CRITERIA["weight_shift"]["advice_good"] if weight_score >= 15
                  else CRITERIA["weight_shift"]["advice_warn"],
    })

    # === 5. 振り切り（フォロースルー） ===
    follow_score = 0
    follow_frame = min(peak + 10, end)
    follow_lm = landmarks_history.get(follow_frame)
    if follow_lm:
        for s, e, w in [(12, 14, 16), (11, 13, 15)]:
            if all(follow_lm[i][3] > 0.5 for i in (s, e, w)):
                angle = calc_angle(
                    (follow_lm[s][0], follow_lm[s][1]),
                    (follow_lm[e][0], follow_lm[e][1]),
                    (follow_lm[w][0], follow_lm[w][1]),
                )
                if angle > 150:
                    follow_score = CRITERIA["follow_through"]["max_score"]
                elif angle > 120:
                    follow_score = int(CRITERIA["follow_through"]["max_score"] * 0.6)
                else:
                    follow_score = int(CRITERIA["follow_through"]["max_score"] * 0.3)
                break

    details.append({
        "name": CRITERIA["follow_through"]["name"],
        "score": follow_score,
        "max": CRITERIA["follow_through"]["max_score"],
        "status": "good" if follow_score >= 12 else "warning" if follow_score >= 8 else "bad",
        "advice": CRITERIA["follow_through"]["advice_good"] if follow_score >= 12
                  else CRITERIA["follow_through"]["advice_warn"],
    })

    # === 6. 頭の安定 ===
    head_score = 0
    nose_positions = []
    for f in range(start, min(end + 1, peak + 5)):
        lm = landmarks_history.get(f)
        if lm and lm[0][3] > 0.5:
            nose_positions.append((lm[0][0], lm[0][1]))

    if len(nose_positions) >= 3:
        xs = [p[0] for p in nose_positions]
        ys = [p[1] for p in nose_positions]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        head_movement = np.sqrt(x_range**2 + y_range**2)

        if head_movement < 0.05:
            head_score = CRITERIA["head_stability"]["max_score"]
        elif head_movement < 0.10:
            head_score = int(CRITERIA["head_stability"]["max_score"] * 0.6)
        else:
            head_score = int(CRITERIA["head_stability"]["max_score"] * 0.3)

    details.append({
        "name": CRITERIA["head_stability"]["name"],
        "score": head_score,
        "max": CRITERIA["head_stability"]["max_score"],
        "status": "good" if head_score >= 8 else "warning" if head_score >= 5 else "bad",
        "advice": CRITERIA["head_stability"]["advice_good"] if head_score >= 8
                  else CRITERIA["head_stability"]["advice_warn"],
    })

    # === 総合スコア ===
    total = sum(d["score"] for d in details)

    if total >= 85:
        grade = "S"
    elif total >= 70:
        grade = "A"
    elif total >= 55:
        grade = "B"
    elif total >= 40:
        grade = "C"
    else:
        grade = "D"

    # サマリー生成
    good_items = [d["name"] for d in details if d["status"] == "good"]
    warn_items = [d["name"] for d in details if d["status"] != "good"]

    summary_parts = []
    if good_items:
        summary_parts.append(f"{'・'.join(good_items)}が良いです！")
    if warn_items:
        summary_parts.append(f"{'・'.join(warn_items)}を意識して練習しましょう。")

    return {
        "total_score": total,
        "grade": grade,
        "details": details,
        "summary": " ".join(summary_parts),
    }
