"""動画の読み込み・書き出しユーティリティ"""

import cv2
import tempfile
import os


class VideoReader:
    """動画ファイルの読み込みクラス"""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, frame_idx):
        """指定フレームを取得"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def iter_frames(self, start=0, end=None):
        """フレームをイテレート"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        end = end or self.total_frames

        for i in range(start, end):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield i, frame

    @property
    def duration_sec(self):
        if self.fps > 0:
            return self.total_frames / self.fps
        return 0

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def save_uploaded_video(uploaded_file):
    """Streamlitのアップロードファイルを一時ファイルに保存

    Returns:
        一時ファイルのパス
    """
    suffix = os.path.splitext(uploaded_file.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def frame_to_jpeg(frame, quality=85):
    """フレームをJPEGバイトに変換"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    return buffer.tobytes()
