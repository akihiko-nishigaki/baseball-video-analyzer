"""ピッチングフォームの総合評価・肩肘負担チェック

少年野球では怪我予防が最も重要。
肩・肘の負担を数値化し、危険なフォームを早期発見する。
"""

import numpy as np
from .angle_analyzer import calc_angle


# ─── 怪我リスク判定基準 ───
# 少年野球の医学的知見に基づく参考値
INJURY_RISK_CRITERIA = {
    "elbow_valgus": {
        "name": "肘の外反ストレス",
        "description": "肘が外側に開きすぎると内側側副靱帯に負担がかかります",
        "safe_range": (140, 180),     # 肘角度が安全な範囲
        "warning_range": (120, 140),  # 注意
        "danger_below": 120,          # 危険
    },
    "shoulder_abduction": {
        "name": "肩の外転角度",
        "description": "肩が水平以上に上がりすぎるとインピンジメントのリスク",
        "safe_range": (70, 100),
        "warning_range_high": 110,
        "warning_range_low": 60,
    },
    "elbow_height": {
        "name": "肘の高さ",
        "description": "リリース時に肘が肩より下がると肩への負担増",
        "ideal": "肩と同じ高さ or やや上",
    },
    "trunk_rotation_timing": {
        "name": "体の開きタイミング",
        "description": "着地前に体が開くと腕に頼った投げ方になり負担増",
    },
    "stride_length": {
        "name": "ストライド長",
        "description": "適切なストライドで下半身のエネルギーを活用",
        "ideal_ratio": (0.7, 0.9),  # 身長比
    },
}


# ─── 評価基準 ───
PITCHING_CRITERIA = {
    "elbow_safety": {
        "name": "肘の安全性",
        "max_score": 25,
        "advice_good": "肘に過度な負担がかかっていません",
        "advice_warn": "肘への負担に注意が必要です",
        "advice_bad": "⚠️ 肘に大きな負担がかかっています。フォーム改善を推奨します",
    },
    "shoulder_safety": {
        "name": "肩の安全性",
        "max_score": 25,
        "advice_good": "肩の使い方が安全です",
        "advice_warn": "肩への負担に注意しましょう",
        "advice_bad": "⚠️ 肩への負担が大きいです",
    },
    "body_usage": {
        "name": "体の使い方",
        "max_score": 20,
        "advice_good": "下半身をしっかり使えています",
        "advice_warn": "もっと下半身を使って投げましょう",
    },
    "stride": {
        "name": "ストライド",
        "max_score": 15,
        "advice_good": "適切なストライドです",
        "advice_warn": "ストライドを調整しましょう",
    },
    "follow_through": {
        "name": "フォロースルー",
        "max_score": 15,
        "advice_good": "体全体を使って投げ切れています",
        "advice_warn": "最後まで体を使い切りましょう",
    },
}


def check_elbow_safety(landmarks_history, pitch, arm="right"):
    """肘の安全性チェック

    Returns:
        {
            "score": int,
            "risk_level": "safe" | "warning" | "danger",
            "details": [str],
            "elbow_angles": [(frame, angle), ...],
        }
    """
    start, end, release, _ = pitch
    elbow_idx = 14 if arm == "right" else 13
    shoulder_idx = 12 if arm == "right" else 11
    wrist_idx = 16 if arm == "right" else 15

    elbow_angles = []
    min_angle = 180
    min_angle_frame = release

    for f in range(start, end + 1):
        lm = landmarks_history.get(f)
        if lm is None:
            continue
        s, e, w = lm[shoulder_idx], lm[elbow_idx], lm[wrist_idx]
        if all(pt[3] > 0.5 for pt in (s, e, w)):
            angle = calc_angle((s[0], s[1]), (e[0], e[1]), (w[0], w[1]))
            elbow_angles.append((f, angle))
            if angle < min_angle:
                min_angle = angle
                min_angle_frame = f

    details = []
    risk_level = "safe"
    score = PITCHING_CRITERIA["elbow_safety"]["max_score"]

    if min_angle < 120:
        risk_level = "danger"
        score = int(score * 0.2)
        details.append(f"肘の最小角度が{min_angle:.0f}°（フレーム{min_angle_frame}）— 外反ストレスが大きい")
    elif min_angle < 140:
        risk_level = "warning"
        score = int(score * 0.6)
        details.append(f"肘の最小角度が{min_angle:.0f}°— やや負担あり")
    else:
        details.append(f"肘の角度は安全範囲内（最小{min_angle:.0f}°）")

    # リリース時の肘角度
    release_lm = landmarks_history.get(release)
    if release_lm:
        s, e, w = release_lm[shoulder_idx], release_lm[elbow_idx], release_lm[wrist_idx]
        if all(pt[3] > 0.5 for pt in (s, e, w)):
            release_angle = calc_angle((s[0], s[1]), (e[0], e[1]), (w[0], w[1]))
            if release_angle < 150:
                details.append(f"リリース時肘角度{release_angle:.0f}°— 肘が曲がり気味")
                score = max(0, score - 5)

    return {
        "score": score,
        "risk_level": risk_level,
        "details": details,
        "elbow_angles": elbow_angles,
    }


def check_shoulder_safety(landmarks_history, pitch, arm="right"):
    """肩の安全性チェック"""
    start, end, release, _ = pitch
    shoulder_idx = 12 if arm == "right" else 11
    elbow_idx = 14 if arm == "right" else 13
    hip_idx = 24 if arm == "right" else 23

    details = []
    risk_level = "safe"
    score = PITCHING_CRITERIA["shoulder_safety"]["max_score"]

    # リリース時の肘の高さチェック
    release_lm = landmarks_history.get(release)
    if release_lm:
        shoulder = release_lm[shoulder_idx]
        elbow = release_lm[elbow_idx]

        if shoulder[3] > 0.5 and elbow[3] > 0.5:
            # Y座標は下が大きい。肘が肩より上 = 肘のY < 肩のY
            height_diff = shoulder[1] - elbow[1]

            if height_diff > 0.02:
                details.append("リリース時に肘が肩より上 — 良い位置です")
            elif height_diff > -0.02:
                details.append("リリース時に肘が肩とほぼ同じ高さ — 問題ありません")
            else:
                risk_level = "warning"
                score = int(score * 0.5)
                details.append("⚠️ リリース時に肘が肩より下がっている — 肩への負担が大きい")

    # 肩の外転角度チェック（アームコッキング時）
    for f in range(start, release):
        lm = landmarks_history.get(f)
        if lm is None:
            continue
        s, e, h = lm[shoulder_idx], lm[elbow_idx], lm[hip_idx]
        if all(pt[3] > 0.5 for pt in (s, e, h)):
            shoulder_angle = calc_angle((e[0], e[1]), (s[0], s[1]), (h[0], h[1]))
            if shoulder_angle > 110:
                if risk_level == "safe":
                    risk_level = "warning"
                    score = int(score * 0.6)
                details.append(f"肩の外転が大きい（{shoulder_angle:.0f}°, F{f}）— インピンジメントに注意")
                break

    if not details:
        details.append("肩の使い方は安全範囲内です")

    return {
        "score": score,
        "risk_level": risk_level,
        "details": details,
    }


def check_body_usage(landmarks_history, pitch, arm="right"):
    """体の使い方チェック（体の開き、下半身の使い方）"""
    start, end, release, _ = pitch
    score = PITCHING_CRITERIA["body_usage"]["max_score"]
    details = []

    # ストライド着地前の体の開きチェック
    hip_rotation_at_stride = None
    for f in range(start - 5, release):
        lm = landmarks_history.get(f)
        if lm is None:
            continue
        ls = lm[11]
        rs = lm[12]
        if ls[3] > 0.5 and rs[3] > 0.5:
            dx = rs[0] - ls[0]
            dy = rs[1] - ls[1]
            rot = np.degrees(np.arctan2(abs(dy), abs(dx)))

            # 早い段階で体が開いている（肩が正面を向いている）
            if f < start and rot > 25:
                details.append("投球前に体が開き気味 — 下半身主導を意識しましょう")
                score = max(0, score - 8)
                break

    # 腰→肩の回旋順序チェック（理想: 腰が先に回る）
    hip_rot_data = []
    shoulder_rot_data = []
    for f in range(start, release + 1):
        lm = landmarks_history.get(f)
        if lm is None:
            continue
        lh, rh = lm[23], lm[24]
        ls, rs = lm[11], lm[12]
        if all(pt[3] > 0.5 for pt in (lh, rh, ls, rs)):
            hip_dx = rh[0] - lh[0]
            hip_rot = abs(hip_dx)
            shoulder_dx = rs[0] - ls[0]
            shoulder_rot = abs(shoulder_dx)
            hip_rot_data.append((f, hip_rot))
            shoulder_rot_data.append((f, shoulder_rot))

    if len(hip_rot_data) >= 3 and len(shoulder_rot_data) >= 3:
        # 腰と肩の最大回旋フレームを比較
        hip_peak_f = max(hip_rot_data, key=lambda x: x[1])[0]
        shoulder_peak_f = max(shoulder_rot_data, key=lambda x: x[1])[0]
        if hip_peak_f <= shoulder_peak_f:
            details.append("腰→肩の順に回旋 — 良い連動です")
        else:
            details.append("肩が先に回っている — 腰主導を意識しましょう")
            score = max(0, score - 6)

    if not details:
        details.append("体の使い方は概ね良好です")

    return {
        "score": score,
        "details": details,
    }


def check_stride_length(landmarks_history, pitch, arm="right"):
    """ストライド長チェック"""
    _, _, release, _ = pitch
    score = PITCHING_CRITERIA["stride"]["max_score"]
    details = []

    # リリース時の足の距離
    lm = landmarks_history.get(release)
    if lm:
        la = lm[27]  # 左足首
        ra = lm[28]  # 右足首
        nose = lm[0]

        if la[3] > 0.5 and ra[3] > 0.5 and nose[3] > 0.5:
            stride_dist = abs(la[0] - ra[0])
            # 身長推定
            ankle_y = max(la[1], ra[1])
            body_height = ankle_y - nose[1]

            if body_height > 0.1:
                stride_ratio = stride_dist / body_height
                if 0.7 <= stride_ratio <= 0.95:
                    details.append(f"ストライド長: 身長の{stride_ratio*100:.0f}% — 適切です")
                elif stride_ratio < 0.6:
                    score = int(score * 0.5)
                    details.append(f"ストライド長: 身長の{stride_ratio*100:.0f}% — もう少し大きく踏み出しましょう")
                elif stride_ratio > 1.0:
                    score = int(score * 0.6)
                    details.append(f"ストライド長: 身長の{stride_ratio*100:.0f}% — 大きすぎると制球が乱れます")
                else:
                    score = int(score * 0.8)
                    details.append(f"ストライド長: 身長の{stride_ratio*100:.0f}%")

    if not details:
        details.append("ストライド長の計測ができませんでした")

    return {
        "score": score,
        "details": details,
    }


def check_follow_through(landmarks_history, pitch, fps, arm="right"):
    """フォロースルーチェック"""
    _, end, release, _ = pitch
    score = PITCHING_CRITERIA["follow_through"]["max_score"]
    details = []

    wrist_idx = 16 if arm == "right" else 15
    shoulder_idx = 12 if arm == "right" else 11

    # リリース後の腕の減速（急停止していないか）
    follow_frames = min(end, release + int(fps * 0.3) if fps > 0 else release + 10)
    wrist_positions = []

    for f in range(release, follow_frames + 1):
        lm = landmarks_history.get(f)
        if lm and lm[wrist_idx][3] > 0.5:
            wrist_positions.append((f, lm[wrist_idx][0], lm[wrist_idx][1]))

    if len(wrist_positions) >= 3:
        # 手首がリリース後も動き続けているか
        total_movement = 0
        for i in range(1, len(wrist_positions)):
            dx = wrist_positions[i][1] - wrist_positions[i-1][1]
            dy = wrist_positions[i][2] - wrist_positions[i-1][2]
            total_movement += np.sqrt(dx**2 + dy**2)

        if total_movement > 0.05:
            details.append("リリース後もしっかり腕を振り切れています")
        else:
            score = int(score * 0.5)
            details.append("リリース後に腕が急停止 — 体全体で減速しましょう")
    else:
        details.append("フォロースルーの分析に十分なデータがありません")

    # 体が前に倒れているか（投げ切り）
    follow_lm = landmarks_history.get(follow_frames)
    if follow_lm:
        nose = follow_lm[0]
        hip = follow_lm[24 if arm == "right" else 23]
        if nose[3] > 0.5 and hip[3] > 0.5:
            # 頭が腰より前にある = 前傾
            forward_lean = nose[1] - hip[1]
            if forward_lean > 0.05:
                details.append("体が前に倒れている — 良いフォロースルーです")
            else:
                details.append("もう少し体を前に倒して投げ切りましょう")
                score = max(0, score - 3)

    return {
        "score": score,
        "details": details,
    }


def evaluate_pitching(landmarks_history, pitch, fps, arm="right"):
    """ピッチングフォームの総合評価

    Returns:
        evaluation: {
            "total_score": int,
            "grade": str,
            "injury_risk": "low" | "medium" | "high",
            "details": [...],
            "summary": str,
            "injury_warnings": [str],
        }
    """
    # 各チェック実行
    elbow = check_elbow_safety(landmarks_history, pitch, arm)
    shoulder = check_shoulder_safety(landmarks_history, pitch, arm)
    body = check_body_usage(landmarks_history, pitch, arm)
    stride = check_stride_length(landmarks_history, pitch, arm)
    follow = check_follow_through(landmarks_history, pitch, fps, arm)

    # 詳細リスト
    details = []
    injury_warnings = []

    # 肘の安全性
    status = "good" if elbow["risk_level"] == "safe" else "warning" if elbow["risk_level"] == "warning" else "bad"
    criteria = PITCHING_CRITERIA["elbow_safety"]
    details.append({
        "name": criteria["name"],
        "score": elbow["score"],
        "max": criteria["max_score"],
        "status": status,
        "advice": criteria[f"advice_{status}"] if f"advice_{status}" in criteria else criteria["advice_warn"],
        "sub_details": elbow["details"],
    })
    if elbow["risk_level"] in ("warning", "danger"):
        injury_warnings.extend(elbow["details"])

    # 肩の安全性
    status = "good" if shoulder["risk_level"] == "safe" else "warning"
    criteria = PITCHING_CRITERIA["shoulder_safety"]
    details.append({
        "name": criteria["name"],
        "score": shoulder["score"],
        "max": criteria["max_score"],
        "status": status,
        "advice": criteria[f"advice_{status}"] if f"advice_{status}" in criteria else criteria["advice_warn"],
        "sub_details": shoulder["details"],
    })
    if shoulder["risk_level"] != "safe":
        injury_warnings.extend(shoulder["details"])

    # 体の使い方
    criteria = PITCHING_CRITERIA["body_usage"]
    b_status = "good" if body["score"] >= 15 else "warning"
    details.append({
        "name": criteria["name"],
        "score": body["score"],
        "max": criteria["max_score"],
        "status": b_status,
        "advice": criteria["advice_good"] if b_status == "good" else criteria["advice_warn"],
        "sub_details": body["details"],
    })

    # ストライド
    criteria = PITCHING_CRITERIA["stride"]
    s_status = "good" if stride["score"] >= 12 else "warning"
    details.append({
        "name": criteria["name"],
        "score": stride["score"],
        "max": criteria["max_score"],
        "status": s_status,
        "advice": criteria["advice_good"] if s_status == "good" else criteria["advice_warn"],
        "sub_details": stride["details"],
    })

    # フォロースルー
    criteria = PITCHING_CRITERIA["follow_through"]
    f_status = "good" if follow["score"] >= 12 else "warning"
    details.append({
        "name": criteria["name"],
        "score": follow["score"],
        "max": criteria["max_score"],
        "status": f_status,
        "advice": criteria["advice_good"] if f_status == "good" else criteria["advice_warn"],
        "sub_details": follow["details"],
    })

    # 総合スコア
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

    # 怪我リスク総合判定
    if elbow["risk_level"] == "danger" or shoulder["risk_level"] == "danger":
        injury_risk = "high"
    elif elbow["risk_level"] == "warning" or shoulder["risk_level"] == "warning":
        injury_risk = "medium"
    else:
        injury_risk = "low"

    # サマリー
    good_items = [d["name"] for d in details if d["status"] == "good"]
    warn_items = [d["name"] for d in details if d["status"] != "good"]

    summary_parts = []
    if good_items:
        summary_parts.append(f"{'・'.join(good_items)}は良好です。")
    if warn_items:
        summary_parts.append(f"{'・'.join(warn_items)}に改善の余地があります。")
    if injury_risk == "high":
        summary_parts.append("⚠️ 怪我のリスクが高いフォームです。早めにフォーム改善に取り組みましょう。")
    elif injury_risk == "medium":
        summary_parts.append("肩・肘への負担にやや注意が必要です。")

    return {
        "total_score": total,
        "grade": grade,
        "injury_risk": injury_risk,
        "details": details,
        "summary": " ".join(summary_parts),
        "injury_warnings": injury_warnings,
        "elbow_angles": elbow.get("elbow_angles", []),
    }
