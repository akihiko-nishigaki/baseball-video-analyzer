"""バッティングフォームチェック・連続写真生成"""

import cv2
import numpy as np

from .angle_analyzer import calc_angle


def check_batting_form(landmarks_history, swing, rotation_history):
    """バッティングフォームの6項目チェック

    Args:
        landmarks_history: 全フレームのlandmarksデータ
        swing: (start, end, peak, peak_speed) タプル
        rotation_history: 各フレームの肩回旋角度リスト

    Returns:
        list of dict: [{name, value, judgement, detail}, ...]
    """
    start, end, peak, _ = swing
    results = []

    # --- 1. 構え（スタンス幅） ---
    # スイング開始付近のフレームで足首間距離/肩幅を計算
    stance_frame = max(0, start - 5)
    lm = landmarks_history.get(stance_frame)
    if lm and all(lm[i][3] > 0.5 for i in (11, 12, 27, 28)):
        ankle_dist = abs(lm[27][0] - lm[28][0])
        shoulder_width = abs(lm[11][0] - lm[12][0])
        if shoulder_width > 0.01:
            ratio = ankle_dist / shoulder_width
            if 0.8 <= ratio <= 1.5:
                judgement = "適切"
            elif ratio < 0.8:
                judgement = "狭い"
            else:
                judgement = "広い"
            results.append({
                "name": "構え（スタンス幅）",
                "value": f"肩幅の{ratio:.2f}倍",
                "judgement": judgement,
                "detail": "足首間距離÷肩幅（0.8〜1.5倍が適切）",
            })
    else:
        results.append({
            "name": "構え（スタンス幅）",
            "value": "-",
            "judgement": "検出不可",
            "detail": "関節が十分に検出できませんでした",
        })

    # --- 2. テイクバック（手首の高さ vs 鼻の高さ） ---
    takeback_frame = max(0, start - 3)
    lm = landmarks_history.get(takeback_frame)
    if lm and lm[0][3] > 0.5 and lm[16][3] > 0.5:
        nose_y = lm[0][1]
        wrist_y = lm[16][1]
        # Y軸は下が大きいので、手首Yが鼻Yに近い = 顎付近
        diff = abs(wrist_y - nose_y)
        if diff < 0.15:
            judgement = "適切"
        elif wrist_y < nose_y:
            judgement = "高すぎ"
        else:
            judgement = "低い"
        results.append({
            "name": "テイクバック",
            "value": f"鼻との差: {diff:.3f}",
            "judgement": judgement,
            "detail": "手首が顎の高さ付近にあるか（差0.15未満が適切）",
        })
    else:
        results.append({
            "name": "テイクバック",
            "value": "-",
            "judgement": "検出不可",
            "detail": "関節が十分に検出できませんでした",
        })

    # --- 3. 体の開き ---
    opening = detect_body_opening_timing(rotation_history, swing)
    if opening:
        results.append({
            "name": "体の開き",
            "value": f"インパクト{opening['frames_before']}F前",
            "judgement": opening["judgement"],
            "detail": opening["detail"],
        })
    else:
        results.append({
            "name": "体の開き",
            "value": "-",
            "judgement": "検出不可",
            "detail": "肩の回旋データが不十分です",
        })

    # --- 4. インパクト（肘角度） ---
    lm = landmarks_history.get(peak)
    if lm:
        elbow_checked = False
        for side, (s_idx, e_idx, w_idx) in [("右肘", (12, 14, 16)), ("左肘", (11, 13, 15))]:
            if all(lm[i][3] > 0.5 for i in (s_idx, e_idx, w_idx)):
                angle = calc_angle(
                    (lm[s_idx][0], lm[s_idx][1]),
                    (lm[e_idx][0], lm[e_idx][1]),
                    (lm[w_idx][0], lm[w_idx][1]),
                )
                if angle >= 140:
                    judgement = "伸びている"
                elif angle >= 120:
                    judgement = "やや曲がり"
                else:
                    judgement = "曲がりすぎ"
                results.append({
                    "name": f"インパクト（{side}角度）",
                    "value": f"{angle:.1f}°",
                    "judgement": judgement,
                    "detail": "140度以上=しっかり伸びている",
                })
                elbow_checked = True
        if not elbow_checked:
            results.append({
                "name": "インパクト（肘角度）",
                "value": "-",
                "judgement": "検出不可",
                "detail": "関節が十分に検出できませんでした",
            })

    # --- 5. フォロースルー（体重移動） ---
    follow_frame = min(peak + 5, end)
    lm = landmarks_history.get(follow_frame)
    if lm and all(lm[i][3] > 0.5 for i in (23, 24, 27, 28)):
        hip_center_x = (lm[23][0] + lm[24][0]) / 2
        left_ankle_x = lm[27][0]
        right_ankle_x = lm[28][0]
        front_ankle_x = min(left_ankle_x, right_ankle_x)
        rear_ankle_x = max(left_ankle_x, right_ankle_x)
        foot_span = rear_ankle_x - front_ankle_x
        if foot_span > 0.01:
            shift = (hip_center_x - front_ankle_x) / foot_span
            if shift < 0.45:
                judgement = "前足寄り（体重移動OK）"
            elif shift < 0.55:
                judgement = "中央"
            else:
                judgement = "後ろ足寄り（体重が残っている）"
            results.append({
                "name": "フォロースルー",
                "value": f"重心位置: {shift:.2f}",
                "judgement": judgement,
                "detail": "腰中心が前足寄り=体重移動できている",
            })
    else:
        results.append({
            "name": "フォロースルー",
            "value": "-",
            "judgement": "検出不可",
            "detail": "関節が十分に検出できませんでした",
        })

    # --- 6. 頭の安定性 ---
    head = calc_head_stability(landmarks_history, swing)
    if head:
        if head["stable"]:
            judgement = "安定"
        else:
            judgement = "ブレあり"
        results.append({
            "name": "頭の安定性",
            "value": f"X偏差:{head['std_x']:.4f} Y偏差:{head['std_y']:.4f}",
            "judgement": judgement,
            "detail": "鼻座標の標準偏差（0.02未満=安定）",
        })
    else:
        results.append({
            "name": "頭の安定性",
            "value": "-",
            "judgement": "検出不可",
            "detail": "鼻の検出データが不十分です",
        })

    return results


def calc_head_stability(landmarks_history, swing):
    """頭の安定性を計算（鼻の位置追跡）

    Args:
        landmarks_history: 全フレームのlandmarksデータ
        swing: (start, end, peak, peak_speed) タプル

    Returns:
        dict: {std_x, std_y, stable, nose_positions} or None
    """
    start, end, _, _ = swing
    nose_xs = []
    nose_ys = []

    for f in range(start, end + 1):
        lm = landmarks_history.get(f)
        if lm and lm[0][3] > 0.5:
            nose_xs.append(lm[0][0])
            nose_ys.append(lm[0][1])

    if len(nose_xs) < 3:
        return None

    std_x = float(np.std(nose_xs))
    std_y = float(np.std(nose_ys))
    stable = std_x < 0.02 and std_y < 0.02

    return {
        "std_x": std_x,
        "std_y": std_y,
        "stable": stable,
        "nose_positions": list(zip(nose_xs, nose_ys)),
    }


def detect_body_opening_timing(rotation_history, swing):
    """体の開きタイミングを判定

    rotation_historyから肩が開くタイミングを検出し、
    インパクト何フレーム前に開いたかで判定。

    Args:
        rotation_history: 各フレームの肩回旋角度リスト
        swing: (start, end, peak, peak_speed) タプル

    Returns:
        dict: {frames_before, judgement, detail} or None
    """
    start, end, peak, _ = swing

    if not rotation_history or peak >= len(rotation_history):
        return None

    # スイング区間の回旋データを取得
    rot_data = []
    for f in range(start, peak + 1):
        if f < len(rotation_history) and rotation_history[f] is not None:
            rot_data.append((f, rotation_history[f]))

    if len(rot_data) < 3:
        return None

    # 肩が「開いた」= 回旋角度が急激に変化するポイント
    # 回旋角度の変化率を計算
    opening_frame = None
    threshold = 2.0  # 角度変化率の閾値

    for i in range(1, len(rot_data)):
        prev_f, prev_rot = rot_data[i - 1]
        curr_f, curr_rot = rot_data[i]
        delta = abs(curr_rot - prev_rot)
        if delta > threshold:
            opening_frame = curr_f
            break

    if opening_frame is None:
        # 変化が緩やかな場合は中間点を採用
        opening_frame = rot_data[len(rot_data) // 2][0]

    frames_before = peak - opening_frame

    if frames_before <= 2:
        judgement = "遅い"
        detail = "肩の開きがインパクト直前で、力が伝わりにくい"
    elif frames_before <= 6:
        judgement = "適切"
        detail = "インパクトに合わせた良いタイミングで体が開いている"
    else:
        judgement = "早い"
        detail = "肩が早く開きすぎてパワーがロスしている可能性あり"

    return {
        "frames_before": frames_before,
        "judgement": judgement,
        "detail": detail,
    }


def create_sequential_photos(reader, landmarks_history, swing, angle_defs,
                             num_photos=8, cols=4):
    """連続写真を生成（骨格付き等間隔8枚→グリッド画像）

    Args:
        reader: VideoReader インスタンス
        landmarks_history: 全フレームのlandmarksデータ
        swing: (start, end, peak, peak_speed) タプル
        angle_defs: 角度定義dict
        num_photos: 抽出枚数 (default 8)
        cols: 列数 (default 4)

    Returns:
        grid_image: BGR numpy array
    """
    start, end, _, _ = swing

    # スイング前後にマージンを取る
    margin = max(5, (end - start) // 4)
    seq_start = max(0, start - margin)
    seq_end = min(reader.total_frames - 1, end + margin)
    total = seq_end - seq_start

    # 等間隔でフレームインデックスを取得
    if total <= num_photos:
        frame_indices = list(range(seq_start, seq_end + 1))
    else:
        frame_indices = [seq_start + int(i * total / (num_photos - 1)) for i in range(num_photos)]
        frame_indices[-1] = seq_end  # 最後のフレームを確保

    # 各フレームを取得して骨格描画
    images = []
    for fidx in frame_indices:
        frame = reader.get_frame(fidx)
        if frame is None:
            continue
        lm = landmarks_history.get(fidx)
        if lm:
            from .pose_detector import draw_skeleton
            frame = draw_skeleton(frame, lm, angle_defs)
        # フレーム番号を表示
        cv2.putText(frame, f"F{fidx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        images.append(frame)

    if not images:
        return None

    # グリッドに配置
    rows = (len(images) + cols - 1) // cols
    # サムネイルサイズ
    thumb_h = 360
    thumb_w = int(thumb_h * images[0].shape[1] / images[0].shape[0])

    grid_h = thumb_h * rows
    grid_w = thumb_w * cols
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        resized = cv2.resize(img, (thumb_w, thumb_h))
        y0 = r * thumb_h
        x0 = c * thumb_w
        grid[y0:y0 + thumb_h, x0:x0 + thumb_w] = resized

    return grid
