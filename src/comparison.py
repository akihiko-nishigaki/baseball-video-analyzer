"""2動画比較モジュール

過去 vs 現在、お手本 vs 自分の動画を並べて比較。
角度差分・改善点のハイライト表示。
"""

import cv2
import numpy as np


def align_frames(total_a, total_b, sync_a=0, sync_b=0):
    """2動画のフレームを同期するためのマッピングを作成

    sync_a/sync_b を基準に、前後のフレームを揃える。
    例：バッティングならインパクトフレーム同士を同期。

    Args:
        total_a: 動画Aの総フレーム数
        total_b: 動画Bの総フレーム数
        sync_a: 動画Aの基準フレーム
        sync_b: 動画Bの基準フレーム

    Returns:
        mapping: [(frame_a, frame_b), ...] 対応フレームのリスト
    """
    offset = sync_b - sync_a
    mapping = []

    # 同期点を基準に前後を揃える
    min_start = min(sync_a, sync_b - offset if offset >= 0 else sync_a)
    start_a = max(0, sync_a - sync_b) if sync_b < sync_a else 0
    start_b = max(0, sync_b - sync_a) if sync_a < sync_b else 0

    fa, fb = start_a, start_b
    while fa < total_a and fb < total_b:
        mapping.append((fa, fb))
        fa += 1
        fb += 1

    return mapping


def compare_angles(angles_a, angles_b, frame_a, frame_b):
    """2フレームの角度を比較

    Args:
        angles_a: 動画Aの全角度データ {frame: {angle_name: value}}
        angles_b: 動画Bの全角度データ
        frame_a: 動画Aのフレーム番号
        frame_b: 動画Bのフレーム番号

    Returns:
        diffs: [{name, value_a, value_b, diff, status}, ...]
    """
    a = angles_a.get(frame_a, {})
    b = angles_b.get(frame_b, {})

    all_names = sorted(set(list(a.keys()) + list(b.keys())))
    diffs = []

    for name in all_names:
        va = a.get(name)
        vb = b.get(name)

        if va is not None and vb is not None:
            diff = vb - va
            abs_diff = abs(diff)
            if abs_diff < 5:
                status = "same"
            elif abs_diff < 15:
                status = "minor"
            else:
                status = "major"
        else:
            diff = None
            status = "missing"

        diffs.append({
            "name": name,
            "value_a": va,
            "value_b": vb,
            "diff": diff,
            "status": status,
        })

    return diffs


def compare_evaluations(eval_a, eval_b):
    """2つの評価結果を比較してスコア変化を算出

    Args:
        eval_a: evaluate_batting/pitching() の結果（過去）
        eval_b: 同（現在）

    Returns:
        comparison: {
            "score_change": int,
            "grade_change": str,
            "improved": [項目名],
            "declined": [項目名],
            "detail_diffs": [{name, score_a, score_b, change}, ...]
        }
    """
    if eval_a is None or eval_b is None:
        return None

    score_a = eval_a["total_score"]
    score_b = eval_b["total_score"]

    detail_diffs = []
    improved = []
    declined = []

    details_a = {d["name"]: d for d in eval_a["details"]}
    details_b = {d["name"]: d for d in eval_b["details"]}

    all_names = list(dict.fromkeys(
        [d["name"] for d in eval_a["details"]] +
        [d["name"] for d in eval_b["details"]]
    ))

    for name in all_names:
        da = details_a.get(name)
        db = details_b.get(name)
        sa = da["score"] if da else 0
        sb = db["score"] if db else 0
        change = sb - sa

        detail_diffs.append({
            "name": name,
            "score_a": sa,
            "score_b": sb,
            "max": da["max"] if da else (db["max"] if db else 0),
            "change": change,
        })

        if change > 0:
            improved.append(name)
        elif change < 0:
            declined.append(name)

    return {
        "score_change": score_b - score_a,
        "grade_a": eval_a["grade"],
        "grade_b": eval_b["grade"],
        "improved": improved,
        "declined": declined,
        "detail_diffs": detail_diffs,
    }


def create_side_by_side(frame_a, frame_b, target_height=480):
    """2フレームを横並びにした画像を作成

    Args:
        frame_a: 動画Aのフレーム（BGR）
        frame_b: 動画Bのフレーム（BGR）
        target_height: 出力画像の高さ

    Returns:
        combined: 横並びの画像（BGR）
    """
    if frame_a is None and frame_b is None:
        return np.zeros((target_height, target_height * 2, 3), dtype=np.uint8)

    if frame_a is None:
        frame_a = np.zeros_like(frame_b)
    if frame_b is None:
        frame_b = np.zeros_like(frame_a)

    # 高さを揃える
    ha, wa = frame_a.shape[:2]
    hb, wb = frame_b.shape[:2]

    scale_a = target_height / ha
    scale_b = target_height / hb

    resized_a = cv2.resize(frame_a, (int(wa * scale_a), target_height))
    resized_b = cv2.resize(frame_b, (int(wb * scale_b), target_height))

    # 区切り線
    separator = np.full((target_height, 4, 3), (100, 100, 100), dtype=np.uint8)

    combined = np.hstack([resized_a, separator, resized_b])
    return combined


def draw_angle_diff_overlay(frame, landmarks, angle_diffs, side="A"):
    """角度差分を動画フレームに描画

    Args:
        frame: BGRフレーム
        landmarks: フレームのランドマーク
        angle_diffs: compare_angles() の結果
        side: "A" or "B"（どちら側のフレームか）

    Returns:
        frame: 描画済みフレーム
    """
    if landmarks is None or not angle_diffs:
        return frame

    h, w = frame.shape[:2]
    y_offset = 30

    for diff_info in angle_diffs:
        status = diff_info["status"]
        if status == "missing":
            continue

        value = diff_info["value_a"] if side == "A" else diff_info["value_b"]
        if value is None:
            continue

        diff_val = diff_info["diff"]
        if diff_val is None:
            continue

        # 色設定
        if status == "same":
            color = (200, 200, 200)  # グレー（差なし）
        elif status == "minor":
            color = (0, 200, 255)    # 黄色（軽微な差）
        else:
            color = (0, 0, 255)      # 赤（大きな差）

        # 差分表示テキスト
        sign = "+" if diff_val > 0 else ""
        if side == "B":
            label = f"{diff_info['name']}: {value:.0f} ({sign}{diff_val:.0f})"
        else:
            label = f"{diff_info['name']}: {value:.0f}"

        cv2.putText(frame, label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_offset += 22

    return frame


def find_sync_point_batting(swings_a, swings_b, sync_mode="swing_start"):
    """バッティングの同期点を決定

    Args:
        swings_a: 動画Aのスイング検出結果
        swings_b: 動画Bのスイング検出結果
        sync_mode: "swing_start" | "impact" | "swing_end"

    Returns:
        (sync_a, sync_b) or (0, 0) if not found
    """
    if not swings_a or not swings_b:
        return 0, 0

    best_a = max(swings_a, key=lambda s: s[3])
    best_b = max(swings_b, key=lambda s: s[3])

    if sync_mode == "swing_start":
        return best_a[0], best_b[0]  # start frames
    elif sync_mode == "impact":
        return best_a[2], best_b[2]  # peak (impact) frames
    elif sync_mode == "swing_end":
        return best_a[1], best_b[1]  # end frames
    else:
        return best_a[0], best_b[0]


def find_sync_point_pitching(pitches_a, pitches_b, sync_mode="pitch_start"):
    """ピッチングの同期点を決定

    Args:
        pitches_a: 動画Aの投球検出結果
        pitches_b: 動画Bの投球検出結果
        sync_mode: "pitch_start" | "release" | "pitch_end"

    Returns:
        (sync_a, sync_b) or (0, 0) if not found
    """
    if not pitches_a or not pitches_b:
        return 0, 0

    best_a = max(pitches_a, key=lambda p: p[3])
    best_b = max(pitches_b, key=lambda p: p[3])

    if sync_mode == "pitch_start":
        return best_a[0], best_b[0]  # start frames
    elif sync_mode == "release":
        return best_a[2], best_b[2]  # release frames
    elif sync_mode == "pitch_end":
        return best_a[1], best_b[1]  # end frames
    else:
        return best_a[0], best_b[0]


def calc_angle_similarity(angles_a, angles_b, frames_a, frames_b):
    """2動画間の角度類似度を計算（全フレーム平均）

    Args:
        angles_a: 動画Aの全角度 {frame: {name: value}}
        angles_b: 動画Bの全角度
        frames_a: 比較対象のAフレームリスト
        frames_b: 比較対象のBフレームリスト

    Returns:
        similarity: 0.0〜1.0（1.0が完全一致）
        per_angle: {angle_name: similarity}
    """
    if not frames_a or not frames_b:
        return 0.0, {}

    count = min(len(frames_a), len(frames_b))
    angle_sums = {}
    angle_counts = {}

    for i in range(count):
        aa = angles_a.get(frames_a[i], {})
        ab = angles_b.get(frames_b[i], {})

        common = set(aa.keys()) & set(ab.keys())
        for name in common:
            diff = abs(aa[name] - ab[name])
            # 180度を最大として正規化
            score = max(0, 1.0 - diff / 90.0)
            if name not in angle_sums:
                angle_sums[name] = 0.0
                angle_counts[name] = 0
            angle_sums[name] += score
            angle_counts[name] += 1

    per_angle = {}
    for name in angle_sums:
        per_angle[name] = angle_sums[name] / angle_counts[name] if angle_counts[name] > 0 else 0.0

    overall = np.mean(list(per_angle.values())) if per_angle else 0.0
    return float(overall), per_angle
