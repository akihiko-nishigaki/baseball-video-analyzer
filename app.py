"""少年野球フォーム分析ツール v4.0 - Streamlit メインアプリ"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(__file__))

from src.pose_detector import PoseDetector, KEY_LANDMARKS, draw_skeleton
from src.angle_analyzer import (
    BATTING_ANGLES, PITCHING_ANGLES,
    analyze_frame_angles, calc_body_rotation, calc_center_of_gravity,
)
from src.swing_detector import calc_wrist_speed, detect_swings, calc_swing_metrics, calc_weight_shift
from src.phase_detector import detect_batting_phases, get_phase_at_frame, get_phase_checkpoints, BATTING_PHASES
from src.trajectory import draw_wrist_trajectory, draw_bat_path, draw_phase_indicator, calc_swing_arc_angle, draw_ghost_skeletons
from src.form_checker import check_batting_form, calc_head_stability, detect_body_opening_timing, create_sequential_photos
from src.batting_evaluator import evaluate_batting
from src.pitching_detector import (
    calc_throwing_arm_speed, detect_pitch_motion, detect_pitching_phases,
    get_pitching_phase_at_frame, detect_release_point, calc_arm_slot,
    PITCHING_PHASES as PITCHING_PHASE_DEFS,
)
from src.pitching_evaluator import evaluate_pitching
from src.comparison import (
    align_frames, compare_angles,
    create_side_by_side, create_top_bottom,
    find_sync_point_batting, find_sync_point_pitching,
)
from utils.video_utils import VideoReader, save_uploaded_video

# ─── ページ設定 ───
st.set_page_config(
    page_title="少年野球フォーム分析",
    page_icon="⚾",
    layout="wide",
)

# ─── カスタムCSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .grade-S { color: #FFD700; font-size: 3rem; font-weight: bold; }
    .grade-A { color: #4CAF50; font-size: 3rem; font-weight: bold; }
    .grade-B { color: #2196F3; font-size: 3rem; font-weight: bold; }
    .grade-C { color: #FF9800; font-size: 3rem; font-weight: bold; }
    .grade-D { color: #F44336; font-size: 3rem; font-weight: bold; }
    .phase-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .check-good { color: #4CAF50; }
    .check-warn { color: #FF9800; }
    .check-bad  { color: #F44336; }
    .stSlider > div > div { padding-top: 0; }
    /* 動画フレームの高さをビューポートに収める */
    [data-testid="stImage"] img {
        max-height: 70vh;
        width: auto !important;
        object-fit: contain;
    }
    /* コマ送りボタンをスマホでも横並びに強制 */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"]:has(button) {
            flex-wrap: nowrap !important;
            gap: 0.25rem !important;
        }
        [data-testid="stHorizontalBlock"]:has(button) [data-testid="stColumn"] {
            min-width: 0 !important;
            flex: 1 1 0 !important;
        }
        [data-testid="stHorizontalBlock"]:has(button) button {
            padding: 0.25rem 0.4rem !important;
            font-size: 0.75rem !important;
            min-height: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─── セッション状態の初期化 ───
def init_session():
    defaults = {
        "video_path": None,
        "current_frame": 0,
        "is_analyzed": False,
        "all_landmarks": {},
        "all_angles": {},
        "cog_history": [],
        "rotation_history": [],
        # Phase 2
        "wrist_speeds": [],
        "swings": [],
        "phases": [],
        "evaluation": None,
        "weight_data": [],
        "checkpoints": [],
        # Phase 2.5 (form checks)
        "form_checks": None,
        "head_stability": None,
        "body_opening": None,
        "sequential_photo": None,
        # Phase 3 (pitching)
        "pitches": [],
        "pitching_phases": [],
        "pitching_evaluation": None,
        "release_info": None,
        "arm_slot": None,
        "throwing_arm": "right",
        "video_name": None,
        # Phase 4 (comparison)
        "video_path_b": None,
        "video_name_b": None,
        "is_analyzed_b": False,
        "all_landmarks_b": {},
        "all_angles_b": {},
        "wrist_speeds_b": [],
        "swings_b": [],
        "phases_b": [],
        "evaluation_b": None,
        "pitches_b": [],
        "pitching_phases_b": [],
        "pitching_evaluation_b": None,
        "release_info_b": None,
        "arm_slot_b": None,
        "compare_frame": 0,
        "frame_mapping": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── サイドバー ───
st.sidebar.markdown("## ⚾ 少年野球フォーム分析")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "アプリモード",
    ["通常分析", "2動画比較"],
    help="通常分析 or 2動画を並べて比較",
)

st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader(
    "動画A をアップロード" if app_mode == "2動画比較" else "動画をアップロード",
    type=["mp4", "mov", "avi", "mkv"],
    help="スマホで撮影した動画ファイルを選択してください",
    key="upload_a",
)

if uploaded:
    if st.session_state.video_name != uploaded.name:
        video_path = save_uploaded_video(uploaded)
        st.session_state.video_path = video_path
        st.session_state.video_name = uploaded.name
        st.session_state.is_analyzed = False
        st.session_state.all_landmarks = {}
        st.session_state.all_angles = {}
        st.session_state.cog_history = []
        st.session_state.rotation_history = []
        st.session_state.wrist_speeds = []
        st.session_state.swings = []
        st.session_state.phases = []
        st.session_state.evaluation = None
        st.session_state.weight_data = []
        st.session_state.checkpoints = []
        st.session_state.pitches = []
        st.session_state.pitching_phases = []
        st.session_state.pitching_evaluation = None
        st.session_state.release_info = None
        st.session_state.arm_slot = None
        st.session_state.form_checks = None
        st.session_state.head_stability = None
        st.session_state.body_opening = None
        st.session_state.sequential_photo = None
        st.session_state.current_frame = 0

# 動画Bアップロード（比較モード時）
if app_mode == "2動画比較":
    uploaded_b = st.sidebar.file_uploader(
        "動画B をアップロード",
        type=["mp4", "mov", "avi", "mkv"],
        help="比較対象の動画を選択してください",
        key="upload_b",
    )
    if uploaded_b:
        if st.session_state.video_name_b != uploaded_b.name:
            video_path_b = save_uploaded_video(uploaded_b)
            st.session_state.video_path_b = video_path_b
            st.session_state.video_name_b = uploaded_b.name
            st.session_state.is_analyzed_b = False
            st.session_state.all_landmarks_b = {}
            st.session_state.all_angles_b = {}
            st.session_state.wrist_speeds_b = []
            st.session_state.swings_b = []
            st.session_state.phases_b = []
            st.session_state.evaluation_b = None
            st.session_state.pitches_b = []
            st.session_state.pitching_phases_b = []
            st.session_state.pitching_evaluation_b = None
            st.session_state.release_info_b = None
            st.session_state.arm_slot_b = None
            st.session_state.compare_frame = 0
            st.session_state.frame_mapping = []

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "分析モード",
    ["バッティング", "ピッチング"],
    help="分析したいフォームの種類を選択",
)
angle_defs = BATTING_ANGLES if mode == "バッティング" else PITCHING_ANGLES

st.sidebar.markdown("---")
st.sidebar.markdown("### 設定")
detection_conf = st.sidebar.slider("検出精度", 0.3, 1.0, 0.5, 0.1)
show_skeleton = st.sidebar.checkbox("骨格を表示", value=True)
show_angles_on_video = st.sidebar.checkbox("角度を動画上に表示", value=True)

if mode == "ピッチング":
    st.sidebar.markdown("### ピッチング設定")
    throwing_arm = st.sidebar.radio("投げ腕", ["右投げ", "左投げ"])
    st.session_state.throwing_arm = "right" if throwing_arm == "右投げ" else "left"

st.sidebar.markdown("### 軌跡表示")
show_wrist_trail = st.sidebar.checkbox("手首の軌跡", value=True)
if mode == "バッティング":
    show_bat_path = st.sidebar.checkbox("バット軌道（推定）", value=False)
else:
    show_bat_path = False
show_ghost = st.sidebar.checkbox("残像表示（ゴースト）", value=False,
                                  help="過去5フレーム分の骨格を半透明で表示")
show_phase_banner = st.sidebar.checkbox("フェーズ表示", value=True)


# ─── メインエリア ───
st.markdown('<div class="main-header">⚾ 少年野球フォーム分析ツール</div>', unsafe_allow_html=True)
st.caption(f"モード: {mode} ｜ v4.0 バッティング＆ピッチング分析・比較・怪我予防チェック")

if app_mode == "2動画比較":
    # ═══════════════════════════════════════════════
    # 2動画比較モード
    # ═══════════════════════════════════════════════
    if st.session_state.video_path is None or st.session_state.video_path_b is None:
        st.info("👈 サイドバーから動画A・動画Bの両方をアップロードしてください")
        st.markdown("""
        ### 2動画比較の使い方
        1. サイドバーで**分析モード**（バッティング/ピッチング）を選択
        2. **動画A**（過去の動画・お手本）をアップロード
        3. **動画B**（現在の動画・自分の動画）をアップロード
        4. 「比較分析開始」ボタンを押す
        5. インパクト/リリースで自動フレーム同期
        """)
        st.stop()

    # 両動画を読み込み
    try:
        reader_a = VideoReader(st.session_state.video_path)
        reader_b = VideoReader(st.session_state.video_path_b)
    except Exception as e:
        st.error(f"動画の読み込みに失敗: {e}")
        st.stop()

    st.sidebar.markdown("### 動画A情報")
    st.sidebar.text(f"解像度: {reader_a.width}x{reader_a.height}")
    st.sidebar.text(f"FPS: {reader_a.fps:.1f} / フレーム: {reader_a.total_frames}")
    st.sidebar.markdown("### 動画B情報")
    st.sidebar.text(f"解像度: {reader_b.width}x{reader_b.height}")
    st.sidebar.text(f"FPS: {reader_b.fps:.1f} / フレーム: {reader_b.total_frames}")

    # ── 分析実行 ──
    needs_analysis = not st.session_state.is_analyzed or not st.session_state.is_analyzed_b
    if needs_analysis:
        if st.button("🔍 比較分析開始", type="primary", use_container_width=True):
            detector = PoseDetector(min_detection_confidence=detection_conf)
            angle_defs_comp = BATTING_ANGLES if mode == "バッティング" else PITCHING_ANGLES
            arm = st.session_state.throwing_arm

            # --- 動画A分析 ---
            if not st.session_state.is_analyzed:
                progress = st.progress(0, text="動画Aを分析中...")
                all_lm_a = {}
                all_ang_a = {}
                for i, frame in reader_a.iter_frames():
                    lm = detector.detect(frame)
                    all_lm_a[i] = lm
                    all_ang_a[i] = analyze_frame_angles(lm, angle_defs_comp, (reader_a.width, reader_a.height))
                    if i % 5 == 0:
                        progress.progress((i + 1) / reader_a.total_frames * 0.4, text=f"動画A骨格検出中... {i+1}/{reader_a.total_frames}")

                ws_a = calc_wrist_speed(all_lm_a, reader_a.fps, wrist_idx=16)
                sw_a = detect_swings(ws_a, reader_a.fps)
                ev_a = None
                ph_a = []
                pi_a = []
                p_ph_a = []
                p_ev_a = None
                rel_a = None
                as_a = None

                if mode == "バッティング" and sw_a:
                    best = max(sw_a, key=lambda s: s[3])
                    ph_a = detect_batting_phases(all_lm_a, ws_a, best, reader_a.fps)
                    wd_a = calc_weight_shift(all_lm_a, best)
                    ev_a = evaluate_batting(all_lm_a, best, wd_a)
                elif mode == "ピッチング":
                    arm_sp_a = calc_throwing_arm_speed(all_lm_a, reader_a.fps, arm=arm)
                    ws_a = arm_sp_a
                    pi_a = detect_pitch_motion(arm_sp_a, reader_a.fps)
                    if pi_a:
                        best_p = max(pi_a, key=lambda p: p[3])
                        p_ph_a = detect_pitching_phases(all_lm_a, arm_sp_a, best_p, reader_a.fps, arm=arm)
                        rel_a = detect_release_point(all_lm_a, best_p, reader_a.fps, arm=arm)
                        as_a = calc_arm_slot(all_lm_a, best_p[2], arm=arm)
                        p_ev_a = evaluate_pitching(all_lm_a, best_p, reader_a.fps, arm=arm)

                st.session_state.all_landmarks = all_lm_a
                st.session_state.all_angles = all_ang_a
                st.session_state.wrist_speeds = ws_a
                st.session_state.swings = sw_a
                st.session_state.phases = ph_a
                st.session_state.evaluation = ev_a
                st.session_state.pitches = pi_a
                st.session_state.pitching_phases = p_ph_a
                st.session_state.pitching_evaluation = p_ev_a
                st.session_state.release_info = rel_a
                st.session_state.arm_slot = as_a
                st.session_state.is_analyzed = True
                progress.progress(0.45, text="動画A完了")

            # --- 動画B分析 ---
            if not st.session_state.is_analyzed_b:
                progress_b = st.progress(0.45, text="動画Bを分析中...")
                all_lm_b = {}
                all_ang_b = {}
                for i, frame in reader_b.iter_frames():
                    lm = detector.detect(frame)
                    all_lm_b[i] = lm
                    all_ang_b[i] = analyze_frame_angles(lm, angle_defs_comp, (reader_b.width, reader_b.height))
                    if i % 5 == 0:
                        progress_b.progress(0.45 + (i + 1) / reader_b.total_frames * 0.4, text=f"動画B骨格検出中... {i+1}/{reader_b.total_frames}")

                ws_b = calc_wrist_speed(all_lm_b, reader_b.fps, wrist_idx=16)
                sw_b = detect_swings(ws_b, reader_b.fps)
                ev_b = None
                ph_b = []
                pi_b = []
                p_ph_b = []
                p_ev_b = None
                rel_b = None
                as_b = None

                if mode == "バッティング" and sw_b:
                    best = max(sw_b, key=lambda s: s[3])
                    ph_b = detect_batting_phases(all_lm_b, ws_b, best, reader_b.fps)
                    wd_b = calc_weight_shift(all_lm_b, best)
                    ev_b = evaluate_batting(all_lm_b, best, wd_b)
                elif mode == "ピッチング":
                    arm_sp_b = calc_throwing_arm_speed(all_lm_b, reader_b.fps, arm=arm)
                    ws_b = arm_sp_b
                    pi_b = detect_pitch_motion(arm_sp_b, reader_b.fps)
                    if pi_b:
                        best_p = max(pi_b, key=lambda p: p[3])
                        p_ph_b = detect_pitching_phases(all_lm_b, arm_sp_b, best_p, reader_b.fps, arm=arm)
                        rel_b = detect_release_point(all_lm_b, best_p, reader_b.fps, arm=arm)
                        as_b = calc_arm_slot(all_lm_b, best_p[2], arm=arm)
                        p_ev_b = evaluate_pitching(all_lm_b, best_p, reader_b.fps, arm=arm)

                st.session_state.all_landmarks_b = all_lm_b
                st.session_state.all_angles_b = all_ang_b
                st.session_state.wrist_speeds_b = ws_b
                st.session_state.swings_b = sw_b
                st.session_state.phases_b = ph_b
                st.session_state.evaluation_b = ev_b
                st.session_state.pitches_b = pi_b
                st.session_state.pitching_phases_b = p_ph_b
                st.session_state.pitching_evaluation_b = p_ev_b
                st.session_state.release_info_b = rel_b
                st.session_state.arm_slot_b = as_b
                st.session_state.is_analyzed_b = True
                progress_b.progress(0.9, text="動画B完了")

            detector.close()

            # --- 初回フレーム同期（スイング/投球開始基準） ---
            if mode == "バッティング":
                sync_a, sync_b = find_sync_point_batting(
                    st.session_state.swings, st.session_state.swings_b,
                    sync_mode="swing_start")
            else:
                sync_a, sync_b = find_sync_point_pitching(
                    st.session_state.pitches, st.session_state.pitches_b,
                    sync_mode="pitch_start")

            mapping = align_frames(reader_a.total_frames, reader_b.total_frames, sync_a, sync_b)
            st.session_state.frame_mapping = mapping
            st.session_state.compare_frame = 0
            st.session_state.sync_a = sync_a
            st.session_state.sync_b = sync_b

            st.success("分析完了！")
            st.rerun()
        else:
            # プレビュー
            col_pa, col_pb = st.columns(2)
            with col_pa:
                fa = reader_a.get_frame(0)
                if fa is not None:
                    st.image(cv2.cvtColor(fa, cv2.COLOR_BGR2RGB), caption="動画A プレビュー", use_container_width=True)
            with col_pb:
                fb = reader_b.get_frame(0)
                if fb is not None:
                    st.image(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB), caption="動画B プレビュー", use_container_width=True)
            reader_a.close()
            reader_b.close()
            st.stop()

    # ════════════════════════════════════════════════
    # 比較結果表示
    # ════════════════════════════════════════════════

    # --- フレーム同期設定 ---
    st.markdown("---")

    # スイング/投球が検出されているか確認
    has_motion_a = bool(st.session_state.swings) if mode == "バッティング" else bool(st.session_state.pitches)
    has_motion_b = bool(st.session_state.swings_b) if mode == "バッティング" else bool(st.session_state.pitches_b)
    has_sync = has_motion_a and has_motion_b

    if has_sync:
        st.markdown("### 🔄 フレーム同期設定")
        sync_col1, sync_col2 = st.columns([1, 1])

        with sync_col1:
            if mode == "バッティング":
                sync_options = {
                    "スイング開始": "swing_start",
                    "インパクト": "impact",
                    "スイング終了": "swing_end",
                }
            else:
                sync_options = {
                    "投球開始": "pitch_start",
                    "リリース": "release",
                    "投球終了": "pitch_end",
                }
            sync_label = st.radio(
                "同期基準",
                list(sync_options.keys()),
                help="2動画のどのタイミングを合わせるか選択",
                key="sync_mode_radio",
                horizontal=True,
            )
            sync_mode = sync_options[sync_label]

        with sync_col2:
            manual_offset = st.slider(
                "手動オフセット（フレーム）",
                -120, 120, 0,
                help="＋で動画Bを遅らせる、ーで動画Bを早める",
                key="manual_offset",
            )

        # 同期ポイント再計算
        if mode == "バッティング":
            sync_a, sync_b = find_sync_point_batting(
                st.session_state.swings, st.session_state.swings_b,
                sync_mode=sync_mode)
        else:
            sync_a, sync_b = find_sync_point_pitching(
                st.session_state.pitches, st.session_state.pitches_b,
                sync_mode=sync_mode)

        sync_b = sync_b + manual_offset
        mapping = align_frames(reader_a.total_frames, reader_b.total_frames, sync_a, sync_b)

        if not mapping:
            mapping = [(i, i) for i in range(min(reader_a.total_frames, reader_b.total_frames))]

        sync_info_col1, sync_info_col2, sync_info_col3 = st.columns(3)
        with sync_info_col1:
            st.caption(f"動画A 基準フレーム: F{sync_a}")
        with sync_info_col2:
            st.caption(f"動画B 基準フレーム: F{sync_b - manual_offset}" +
                       (f" ({manual_offset:+d})" if manual_offset != 0 else ""))
        with sync_info_col3:
            st.caption(f"比較可能フレーム数: {len(mapping)}")
    else:
        # 動作未検出時はフレーム番号をそのまま1:1対応
        missing = []
        if not has_motion_a:
            missing.append("動画A")
        if not has_motion_b:
            missing.append("動画B")
        st.warning(f"{'・'.join(missing)} でスイング/投球動作が検出されませんでした。フレーム番号でそのまま比較します。")

        # 手動オフセットだけ提供
        manual_offset = st.slider(
            "手動オフセット（フレーム）",
            -120, 120, 0,
            help="＋で動画Bを遅らせる、ーで動画Bを早める",
            key="manual_offset_fallback",
        )
        sync_a, sync_b = 0, manual_offset
        mapping = align_frames(reader_a.total_frames, reader_b.total_frames, sync_a, sync_b)
        if not mapping:
            mapping = [(i, i) for i in range(min(reader_a.total_frames, reader_b.total_frames))]

    # --- 同期フレームビューア ---
    st.markdown("---")
    st.markdown("### 🎥 同期フレームビューア")

    # ジャンプ要求があればスライダー作成前にキーへ反映
    if "_cmp_jump_to" in st.session_state:
        st.session_state.compare_slider = st.session_state._cmp_jump_to
        del st.session_state._cmp_jump_to

    # mappingサイズが変わった場合にcompare_frameをクランプ
    max_idx = len(mapping) - 1
    clamped = min(st.session_state.compare_frame, max_idx)
    if clamped != st.session_state.compare_frame:
        st.session_state.compare_frame = clamped
        st.session_state.compare_slider = clamped

    compare_idx = st.slider(
        "比較フレーム",
        0, max_idx,
        clamped,
        key="compare_slider",
    )
    st.session_state.compare_frame = compare_idx

    frame_a_idx, frame_b_idx = mapping[compare_idx]

    # コマ送りボタン
    cmp_btn_cols = st.columns(5)
    with cmp_btn_cols[0]:
        if st.button("⏮ -10", key="cmp_bk10"):
            st.session_state._cmp_jump_to = max(0, compare_idx - 10)
            st.rerun()
    with cmp_btn_cols[1]:
        if st.button("◀ -1", key="cmp_bk1"):
            st.session_state._cmp_jump_to = max(0, compare_idx - 1)
            st.rerun()
    with cmp_btn_cols[2]:
        st.markdown(f"**A:F{frame_a_idx} / B:F{frame_b_idx}**")
    with cmp_btn_cols[3]:
        if st.button("+1 ▶", key="cmp_fw1"):
            st.session_state._cmp_jump_to = min(len(mapping) - 1, compare_idx + 1)
            st.rerun()
    with cmp_btn_cols[4]:
        if st.button("+10 ⏭", key="cmp_fw10"):
            st.session_state._cmp_jump_to = min(len(mapping) - 1, compare_idx + 10)
            st.rerun()

    # 並べて表示（1枚の合成画像にしてスマホでも横並びを維持）
    fa = reader_a.get_frame(frame_a_idx)
    fb = reader_b.get_frame(frame_b_idx)

    if fa is not None:
        lm_a = st.session_state.all_landmarks.get(frame_a_idx)
        if show_skeleton and lm_a:
            fa = draw_skeleton(fa, lm_a, angle_defs if show_angles_on_video else None)
        if show_wrist_trail:
            fa = draw_wrist_trajectory(fa, st.session_state.all_landmarks, frame_a_idx, trail_length=40)
        # ラベル描画
        cv2.putText(fa, "A", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)

    if fb is not None:
        lm_b = st.session_state.all_landmarks_b.get(frame_b_idx)
        if show_skeleton and lm_b:
            fb = draw_skeleton(fb, lm_b, angle_defs if show_angles_on_video else None)
        if show_wrist_trail:
            fb = draw_wrist_trajectory(fb, st.session_state.all_landmarks_b, frame_b_idx, trail_length=40)
        cv2.putText(fb, "B", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 160, 0), 3, cv2.LINE_AA)

    # 両方縦向き → 横並び、それ以外 → 縦並び
    both_portrait = (reader_a.height > reader_a.width) and (reader_b.height > reader_b.width)
    if both_portrait:
        combined = create_side_by_side(fa, fb)
    else:
        combined = create_top_bottom(fa, fb)
    st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- 角度比較テーブル ---
    angle_diffs = compare_angles(
        st.session_state.all_angles, st.session_state.all_angles_b,
        frame_a_idx, frame_b_idx)

    if angle_diffs:
        st.markdown("#### 📐 現在フレームの角度比較")
        diff_data = []
        for d in angle_diffs:
            va = f"{d['value_a']:.1f}" if d["value_a"] is not None else "-"
            vb = f"{d['value_b']:.1f}" if d["value_b"] is not None else "-"
            if d["diff"] is not None:
                sign = "+" if d["diff"] > 0 else ""
                diff_str = f"{sign}{d['diff']:.1f}"
            else:
                diff_str = "-"
            status_icon = {"same": "=", "minor": "~", "major": "!!", "missing": "?"}
            diff_data.append({
                "角度": d["name"],
                "動画A": va,
                "動画B": vb,
                "差分": diff_str,
                "判定": status_icon.get(d["status"], ""),
            })
        st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)

    # --- 角度推移比較グラフ ---
    if st.session_state.all_angles and st.session_state.all_angles_b:
        st.markdown("---")
        st.markdown("### 📈 角度推移の比較")

        # 全角度名を収集
        angle_names_set = set()
        for angles in st.session_state.all_angles.values():
            angle_names_set.update(angles.keys())
        angle_names_list = sorted(angle_names_set)

        if angle_names_list:
            fig_angles = make_subplots(
                rows=len(angle_names_list), cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=angle_names_list,
            )

            for row_i, aname in enumerate(angle_names_list, 1):
                # 動画Aのデータ
                vals_a = []
                for fa_i, fb_i in mapping:
                    v = st.session_state.all_angles.get(fa_i, {}).get(aname)
                    vals_a.append(v)
                # 動画Bのデータ
                vals_b = []
                for fa_i, fb_i in mapping:
                    v = st.session_state.all_angles_b.get(fb_i, {}).get(aname)
                    vals_b.append(v)

                x_axis = list(range(len(mapping)))

                fig_angles.add_trace(
                    go.Scatter(x=x_axis, y=vals_a, mode="lines",
                               name=f"A: {aname}", line=dict(color="#FF9800", width=2),
                               showlegend=(row_i == 1)),
                    row=row_i, col=1,
                )
                fig_angles.add_trace(
                    go.Scatter(x=x_axis, y=vals_b, mode="lines",
                               name=f"B: {aname}", line=dict(color="#2196F3", width=2),
                               showlegend=(row_i == 1)),
                    row=row_i, col=1,
                )

                # 現在位置
                fig_angles.add_vline(
                    x=compare_idx, line_dash="dash", line_color="white",
                    line_width=1, row=row_i, col=1)

            fig_angles.update_layout(
                height=200 * len(angle_names_list),
                margin=dict(l=40, r=20, t=40, b=40),
                template="plotly_dark",
                legend=dict(orientation="h", y=1.02),
            )
            fig_angles.update_xaxes(title_text="同期フレーム", row=len(angle_names_list), col=1)
            st.plotly_chart(fig_angles, use_container_width=True)

    # フッター（比較モード）
    st.markdown("---")
    st.caption("⚾ 少年野球フォーム分析ツール v4.0 ｜ 2動画比較モード")
    reader_a.close()
    reader_b.close()
    st.stop()


# ═══════════════════════════════════════════════
# 通常分析モード（以下は従来通り）
# ═══════════════════════════════════════════════

if st.session_state.video_path is None:
    st.info("👈 左のサイドバーから動画をアップロードしてください")
    st.markdown("""
    ### 使い方
    1. スマホで**バッティング**または**ピッチング**の動画を撮影
    2. サイドバーの「動画をアップロード」から動画ファイルを選択
    3. 「分析開始」ボタンを押す
    4. スライダーでコマ送り＆角度を確認

    ### v4.0 機能
    #### バッティング
    - スイング自動検出 ＆ フェーズ分割
    - バット軌道表示・総合評価（100点満点）

    #### ピッチング
    - 投球動作の自動検出 ＆ フェーズ分割
    - **リリースポイント検出** ＆ アームスロット判定
    - **肩・肘の負担チェック**（怪我予防）
    - 体の開き・ストライド長の評価

    #### 2動画比較（NEW!）
    - 過去の自分 vs 今の自分
    - お手本動画 vs 自分の動画
    - **同期再生**（インパクト/リリースで自動フレーム合わせ）
    - 角度差分・スコア変化の可視化

    ### 撮影のコツ
    - 全身が映るように（頭からつま先まで）
    - 背景はなるべくシンプルに、スマホは**横向き固定**
    - バッティング: **正面やや斜め前**から
    - ピッチング: **三塁側（右投手）/ 一塁側（左投手）**から
    """)
    st.stop()

# ─── 動画読み込み ───
try:
    reader = VideoReader(st.session_state.video_path)
except Exception as e:
    st.error(f"動画の読み込みに失敗: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 動画情報")
st.sidebar.text(f"解像度: {reader.width}x{reader.height}")
st.sidebar.text(f"FPS: {reader.fps:.1f}")
st.sidebar.text(f"フレーム数: {reader.total_frames}")
st.sidebar.text(f"再生時間: {reader.duration_sec:.1f}秒")

# ─── 分析実行 ───
if not st.session_state.is_analyzed:
    if st.button("🔍 分析開始", type="primary", use_container_width=True):
        detector = PoseDetector(min_detection_confidence=detection_conf)

        progress = st.progress(0, text="動画を分析中...")
        all_landmarks = {}
        all_angles = {}
        cog_history = []
        rotation_history = []

        # Step 1: 骨格検出
        for i, frame in reader.iter_frames():
            landmarks = detector.detect(frame)
            all_landmarks[i] = landmarks
            angles = analyze_frame_angles(landmarks, angle_defs, (reader.width, reader.height))
            all_angles[i] = angles
            cog = calc_center_of_gravity(landmarks)
            cog_history.append(cog)
            rot = calc_body_rotation(landmarks)
            rotation_history.append(rot)

            if i % 5 == 0:
                progress.progress(
                    (i + 1) / reader.total_frames * 0.7,
                    text=f"骨格検出中... {i+1}/{reader.total_frames}"
                )

        detector.close()

        # Step 2: スイング検出 & フェーズ分割
        progress.progress(0.75, text="スイングを検出中...")
        wrist_speeds = calc_wrist_speed(all_landmarks, reader.fps, wrist_idx=16)
        swings = detect_swings(wrist_speeds, reader.fps)

        phases = []
        evaluation = None
        weight_data = []
        checkpoints = []

        # ─── バッティングフォームチェック ───
        form_checks = None
        head_stability = None
        body_opening = None

        if mode == "バッティング" and swings:
            progress.progress(0.78, text="フォームチェック中...")
            best_swing = max(swings, key=lambda s: s[3])
            form_checks = check_batting_form(all_landmarks, best_swing, rotation_history)
            head_stability = calc_head_stability(all_landmarks, best_swing)
            body_opening = detect_body_opening_timing(rotation_history, best_swing)

        # ─── ピッチング分析 or バッティング分析 ───
        pitches = []
        pitching_phases = []
        pitching_evaluation = None
        release_info = None
        arm_slot_val = None

        if mode == "ピッチング":
            arm = st.session_state.throwing_arm
            wrist_idx = 16 if arm == "right" else 15

            progress.progress(0.75, text="投球動作を検出中...")
            arm_speeds = calc_throwing_arm_speed(all_landmarks, reader.fps, arm=arm)
            wrist_speeds = arm_speeds  # グラフ用に保存
            pitches = detect_pitch_motion(arm_speeds, reader.fps)

            if pitches:
                best_pitch = max(pitches, key=lambda p: p[3])

                progress.progress(0.82, text="投球フェーズを分析中...")
                pitching_phases = detect_pitching_phases(
                    all_landmarks, arm_speeds, best_pitch, reader.fps, arm=arm)

                progress.progress(0.88, text="リリースポイントを検出中...")
                release_info = detect_release_point(
                    all_landmarks, best_pitch, reader.fps, arm=arm)
                arm_slot_val = calc_arm_slot(
                    all_landmarks, best_pitch[2], arm=arm)

                progress.progress(0.93, text="肩・肘の安全性をチェック中...")
                pitching_evaluation = evaluate_pitching(
                    all_landmarks, best_pitch, reader.fps, arm=arm)

        progress.progress(1.0, text="完了！")
        progress.empty()

        # セッションに保存
        st.session_state.all_landmarks = all_landmarks
        st.session_state.all_angles = all_angles
        st.session_state.cog_history = cog_history
        st.session_state.rotation_history = rotation_history
        st.session_state.wrist_speeds = wrist_speeds
        st.session_state.swings = swings
        st.session_state.phases = phases
        st.session_state.evaluation = evaluation
        st.session_state.weight_data = weight_data
        st.session_state.checkpoints = checkpoints
        st.session_state.form_checks = form_checks
        st.session_state.head_stability = head_stability
        st.session_state.body_opening = body_opening
        st.session_state.sequential_photo = None
        st.session_state.pitches = pitches
        st.session_state.pitching_phases = pitching_phases
        st.session_state.pitching_evaluation = pitching_evaluation
        st.session_state.release_info = release_info
        st.session_state.arm_slot = arm_slot_val
        st.session_state.is_analyzed = True
        st.rerun()
    else:
        preview_frame = reader.get_frame(0)
        if preview_frame is not None:
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb, caption="プレビュー（分析前）", use_container_width=True)
        st.stop()


# ════════════════════════════════════════════════
# 分析結果表示
# ════════════════════════════════════════════════

swings = st.session_state.swings


# ─── ピッチング総合評価（ピッチングモード時） ───
pitching_eval = st.session_state.pitching_evaluation
pitching_phases = st.session_state.pitching_phases
pitches = st.session_state.pitches

if mode == "ピッチング" and pitching_eval:
    st.markdown("---")

    # 怪我リスクバナー
    risk = pitching_eval["injury_risk"]
    if risk == "high":
        st.error("⚠️ **怪我リスク: 高** — フォーム改善を強く推奨します")
    elif risk == "medium":
        st.warning("⚠️ **怪我リスク: 中** — 肩・肘への負担にやや注意")
    else:
        st.success("✅ **怪我リスク: 低** — 安全なフォームです")

    eval_col1, eval_col2, eval_col3 = st.columns([1, 2, 2])

    with eval_col1:
        grade = pitching_eval["grade"]
        st.markdown(f'<div class="grade-{grade}" style="text-align:center;">{grade}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; font-size:1.5rem;'>"
                    f"<b>{pitching_eval['total_score']}</b>/100点</div>",
                    unsafe_allow_html=True)

    with eval_col2:
        st.markdown("#### 評価詳細")
        for d in pitching_eval["details"]:
            icon = "✅" if d["status"] == "good" else "⚠️" if d["status"] == "warning" else "❌"
            st.markdown(f"{icon} **{d['name']}** {d['score']}/{d['max']}")
            st.progress(int(d["score"] / d["max"] * 100) / 100)

    with eval_col3:
        st.markdown("#### アドバイス")
        st.info(pitching_eval["summary"])

        # リリースポイント情報
        rel = st.session_state.release_info
        if rel:
            st.markdown("#### リリースポイント")
            st.text(f"フレーム: {rel['frame']}")
            st.text(f"肘角度: {rel['elbow_angle']:.1f}°")
            if rel["shoulder_angle"]:
                st.text(f"肩角度: {rel['shoulder_angle']:.1f}°")
            if rel["height_ratio"]:
                st.text(f"リリース高さ: 身長の{rel['height_ratio']*100:.0f}%")

        # アームスロット
        arm_slot_val = st.session_state.arm_slot
        if arm_slot_val is not None:
            if arm_slot_val > 70:
                slot_name = "オーバースロー"
            elif arm_slot_val > 45:
                slot_name = "スリークォーター"
            elif arm_slot_val > 15:
                slot_name = "サイドスロー"
            else:
                slot_name = "アンダースロー"
            st.text(f"アームスロット: {slot_name} ({arm_slot_val:.0f}°)")

    # 怪我リスク詳細
    if pitching_eval["injury_warnings"]:
        st.markdown("---")
        st.markdown("### ⚠️ 怪我リスク詳細")
        for warn in pitching_eval["injury_warnings"]:
            st.markdown(f"- {warn}")

    # 評価サブ詳細
    with st.expander("📋 各項目の詳細チェック結果"):
        for d in pitching_eval["details"]:
            st.markdown(f"**{d['name']}** ({d['score']}/{d['max']})")
            for sd in d.get("sub_details", []):
                st.markdown(f"  - {sd}")
            st.markdown("")

    # 肘角度推移グラフ
    elbow_angles = pitching_eval.get("elbow_angles", [])
    if elbow_angles:
        st.markdown("---")
        st.markdown("### 💪 肘角度の推移（怪我予防チェック）")

        ea_frames = [a[0] for a in elbow_angles]
        ea_values = [a[1] for a in elbow_angles]
        ea_times = [f / reader.fps for f in ea_frames] if reader.fps > 0 else ea_frames

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=ea_times, y=ea_values, mode="lines",
            name="肘角度", line=dict(color="#FF5722", width=2),
        ))

        # 安全ゾーン
        fig_elbow.add_hrect(y0=140, y1=180,
                            fillcolor="rgba(76,175,80,0.15)", line_width=0,
                            annotation_text="安全", annotation_position="right")
        fig_elbow.add_hrect(y0=120, y1=140,
                            fillcolor="rgba(255,193,7,0.15)", line_width=0,
                            annotation_text="注意", annotation_position="right")
        fig_elbow.add_hrect(y0=0, y1=120,
                            fillcolor="rgba(244,67,54,0.15)", line_width=0,
                            annotation_text="危険", annotation_position="right")

        # リリースポイント
        if rel:
            rel_time = rel["frame"] / reader.fps if reader.fps > 0 else rel["frame"]
            fig_elbow.add_vline(x=rel_time, line_dash="dash", line_color="cyan",
                                annotation_text="リリース")

        fig_elbow.update_layout(
            xaxis_title="時間（秒）", yaxis_title="肘角度（度）",
            height=300, template="plotly_dark",
            margin=dict(l=40, r=80, t=30, b=40),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

elif mode == "ピッチング" and not pitches:
    st.warning("投球動作が検出されませんでした。動画にピッチングの動きが含まれているか確認してください。")


# ─── フェーズタイムライン ───
# ピッチングフェーズ
if mode == "ピッチング" and pitching_phases:
    st.markdown("---")
    st.markdown("### 🔄 ピッチングフェーズ")

    phase_cols = st.columns(len(pitching_phases))
    for i, (key, p_start, p_end) in enumerate(pitching_phases):
        info = PITCHING_PHASE_DEFS[key]
        with phase_cols[i]:
            st.markdown(
                f'<div class="phase-badge" style="background:{info["color"]};">'
                f'{info["emoji"]} {info["name"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"F{p_start}-{p_end}")
            if st.button(f"▶ {info['name']}", key=f"pphase_{key}"):
                st.session_state._jump_to = p_start
                st.rerun()




# ─── 動画ビューア ───
st.markdown("---")
st.markdown("### 🎥 フレームビューア")

# ジャンプ要求があればスライダー作成前にキーへ反映
if "_jump_to" in st.session_state:
    st.session_state.frame_slider = st.session_state._jump_to
    del st.session_state._jump_to

col_slider, col_info = st.columns([4, 1])
with col_slider:
    frame_idx = st.slider(
        "フレーム",
        0, reader.total_frames - 1,
        st.session_state.current_frame,
        key="frame_slider",
    )
    st.session_state.current_frame = frame_idx

with col_info:
    time_sec = frame_idx / reader.fps if reader.fps > 0 else 0
    st.metric("時間", f"{time_sec:.2f}秒")

# コマ送りボタン
btn_cols = st.columns(5)
with btn_cols[0]:
    if st.button("⏮ -10"):
        st.session_state._jump_to = max(0, frame_idx - 10)
        st.rerun()
with btn_cols[1]:
    if st.button("◀ -1"):
        st.session_state._jump_to = max(0, frame_idx - 1)
        st.rerun()
with btn_cols[2]:
    st.markdown(f"**{frame_idx} / {reader.total_frames - 1}**")
with btn_cols[3]:
    if st.button("+1 ▶"):
        st.session_state._jump_to = min(reader.total_frames - 1, frame_idx + 1)
        st.rerun()
with btn_cols[4]:
    if st.button("+10 ⏭"):
        st.session_state._jump_to = min(reader.total_frames - 1, frame_idx + 10)
        st.rerun()

# スイング/投球 区間へのジャンプボタン
if mode == "ピッチング" and pitches:
    pitch_cols = st.columns(len(pitches) + 1)
    with pitch_cols[0]:
        st.markdown("**投球:**")
    for i, (p_start, p_end, p_release, p_speed) in enumerate(pitches):
        with pitch_cols[i + 1]:
            if st.button(f"⚾ #{i+1} (F{p_start}-{p_end})", key=f"pitch_jump_{i}"):
                st.session_state._jump_to = p_start
                st.rerun()



# ─── 動画フレーム＋骨格表示 ───
col_video, col_angles = st.columns([3, 1])

with col_video:
    frame = reader.get_frame(frame_idx)
    if frame is not None:
        landmarks = st.session_state.all_landmarks.get(frame_idx)

        # 骨格描画
        if show_skeleton and landmarks:
            angles_to_show = angle_defs if show_angles_on_video else None
            frame = draw_skeleton(frame, landmarks, angles_to_show)

        # 残像（ゴースト）表示
        if show_ghost:
            frame = draw_ghost_skeletons(
                frame, st.session_state.all_landmarks, frame_idx,
                ghost_count=5, ghost_step=3,
            )

        # 手首の軌跡
        if show_wrist_trail:
            frame = draw_wrist_trajectory(
                frame, st.session_state.all_landmarks, frame_idx,
                trail_length=40,
            )

        # バット軌道
        if show_bat_path:
            frame = draw_bat_path(
                frame, st.session_state.all_landmarks, frame_idx,
                trail_length=30,
            )

        # フェーズ表示バナー（ピッチング時のみ）
        if show_phase_banner and mode == "ピッチング" and pitching_phases:
            phase_key, phase_info = get_pitching_phase_at_frame(pitching_phases, frame_idx)
            if phase_key and phase_info:
                progress_ratio = 0
                for pk, ps, pe in pitching_phases:
                    if pk == phase_key:
                        progress_ratio = (frame_idx - ps) / max(1, pe - ps)
                        break
                frame = draw_phase_indicator(frame, phase_key, phase_info, progress_ratio)

        # リリースポイントマーカー（ピッチング時）
        if mode == "ピッチング" and st.session_state.release_info:
            rel = st.session_state.release_info
            if frame_idx == rel["frame"]:
                h_f, w_f = frame.shape[:2]
                rx, ry = int(rel["position"][0] * w_f), int(rel["position"][1] * h_f)
                cv2.circle(frame, (rx, ry), 12, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "RELEASE", (rx + 15, ry - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)

with col_angles:
    # フェーズ表示
    if mode == "ピッチング" and pitching_phases:
        phase_key, phase_info = get_pitching_phase_at_frame(pitching_phases, frame_idx)
        if phase_key and phase_info:
            st.markdown(
                f'<div class="phase-badge" style="background:{phase_info["color"]};">'
                f'{phase_info["emoji"]} {phase_info["name"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("#### 📐 現在の角度")
    angles = st.session_state.all_angles.get(frame_idx, {})
    if angles:
        for name, value in angles.items():
            st.metric(name, f"{value:.1f}°")
    else:
        st.caption("検出なし")

    rot = st.session_state.rotation_history[frame_idx] if frame_idx < len(st.session_state.rotation_history) else None
    if rot is not None:
        st.metric("肩の開き", f"{rot:.1f}°")

    # 頭の安定性メトリック（バッティング時）
    if mode == "バッティング" and st.session_state.head_stability:
        hs = st.session_state.head_stability
        st.markdown("---")
        st.markdown("#### 頭の安定性")
        stability_label = "安定" if hs["stable"] else "ブレあり"
        st.metric("判定", stability_label)
        st.caption(f"X偏差: {hs['std_x']:.4f} / Y偏差: {hs['std_y']:.4f}")

    # ピッチング: 投球内かどうか
    if mode == "ピッチング":
        for p_start, p_end, p_release, _ in pitches:
            if p_start <= frame_idx <= p_end:
                st.success("⚾ 投球動作中")
                if frame_idx == p_release:
                    st.markdown("**🎯 リリース！**")
                break




# ─── フォームチェック（バッティング時） ───
if mode == "バッティング" and st.session_state.form_checks:
    st.markdown("---")
    st.markdown("### 📋 フォームチェック")

    for check in st.session_state.form_checks:
        j = check["judgement"]
        if j in ("適切", "安定", "伸びている", "前足寄り（体重移動OK）"):
            css_class = "check-good"
            icon = "✅"
        elif j in ("検出不可",):
            css_class = ""
            icon = "❓"
        else:
            css_class = "check-warn"
            icon = "⚠️"

        st.markdown(
            f'{icon} **{check["name"]}** — '
            f'<span class="{css_class}">{j}</span> '
            f'（{check["value"]}）',
            unsafe_allow_html=True,
        )
        st.caption(check["detail"])

    # 体の開き詳細
    if st.session_state.body_opening:
        bo = st.session_state.body_opening
        st.markdown(f"**体の開きタイミング:** インパクト{bo['frames_before']}フレーム前 → {bo['judgement']}")
        st.caption(bo["detail"])


# ─── 連続写真 ───
if mode == "バッティング" and swings:
    st.markdown("---")
    st.markdown("### 📸 連続写真")

    if st.button("連続写真を生成", key="gen_seq_photo"):
        best_swing = max(swings, key=lambda s: s[3])
        with st.spinner("連続写真を生成中..."):
            grid = create_sequential_photos(
                reader, st.session_state.all_landmarks, best_swing,
                angle_defs, num_photos=8, cols=4,
            )
            if grid is not None:
                st.session_state.sequential_photo = grid

    if st.session_state.sequential_photo is not None:
        grid_rgb = cv2.cvtColor(st.session_state.sequential_photo, cv2.COLOR_BGR2RGB)
        st.image(grid_rgb, caption="スイング連続写真（骨格付き）", use_container_width=True)

        # ダウンロードボタン
        _, buf = cv2.imencode(".png", st.session_state.sequential_photo)
        st.download_button(
            label="連続写真をダウンロード",
            data=buf.tobytes(),
            file_name="sequential_photos.png",
            mime="image/png",
        )


# ─── 体重移動グラフ ───
weight_data = st.session_state.weight_data
if weight_data:
    st.markdown("---")
    st.markdown("### ⚖️ 体重移動")

    fig_weight = make_subplots(
        rows=1, cols=2,
        subplot_titles=("重心位置 (左右)", "体重配分"),
        column_widths=[0.5, 0.5],
    )

    w_frames = [d[0] for d in weight_data]
    w_times = [f / reader.fps for f in w_frames] if reader.fps > 0 else w_frames
    w_cog_x = [d[1] for d in weight_data]
    w_ratio = [d[2] for d in weight_data]

    fig_weight.add_trace(
        go.Scatter(x=w_times, y=w_cog_x, mode="lines+markers",
                   marker=dict(size=4), name="重心X", line=dict(color="#2196F3")),
        row=1, col=1,
    )

    fig_weight.add_trace(
        go.Scatter(x=w_times, y=w_ratio, mode="lines+markers",
                   marker=dict(size=4), name="体重配分", line=dict(color="#FF9800"),
                   fill="tozeroy", fillcolor="rgba(255,152,0,0.2)"),
        row=1, col=2,
    )

    fig_weight.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=2,
                         annotation_text="中央")

    fig_weight.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_dark",
    )
    fig_weight.update_yaxes(title_text="位置", row=1, col=1)
    fig_weight.update_yaxes(title_text="前足←→後ろ足", range=[0, 1], row=1, col=2)

    st.plotly_chart(fig_weight, use_container_width=True)


# ─── 重心移動（2D） ───
if any(c is not None for c in st.session_state.cog_history):
    st.markdown("### 📍 重心軌跡")
    cog_x = [c[0] if c else None for c in st.session_state.cog_history]
    cog_y = [c[1] if c else None for c in st.session_state.cog_history]

    fig_cog = go.Figure()
    fig_cog.add_trace(go.Scatter(
        x=cog_x, y=cog_y, mode="markers+lines",
        marker=dict(size=4, color=list(range(len(cog_x))),
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="フレーム")),
        line=dict(color="rgba(255,255,255,0.3)", width=1),
        name="重心",
    ))

    if frame_idx < len(cog_x) and cog_x[frame_idx] is not None:
        fig_cog.add_trace(go.Scatter(
            x=[cog_x[frame_idx]], y=[cog_y[frame_idx]],
            mode="markers", marker=dict(size=15, color="red", symbol="x"),
            name="現在",
        ))

    fig_cog.update_layout(
        xaxis_title="左右", yaxis_title="上下",
        yaxis=dict(autorange="reversed"),
        height=300, margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_dark",
    )
    st.plotly_chart(fig_cog, use_container_width=True)


# ─── フッター ───
st.markdown("---")
st.caption("⚾ 少年野球フォーム分析ツール v4.0 ｜ MediaPipe Pose + OpenCV + Streamlit")

reader.close()
