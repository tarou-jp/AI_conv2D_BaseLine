import csv
import cv2
import os
import sys
import numpy as np
from datetime import datetime
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger
from utils.metrics import calculate_accuracy
from utils.config import config

class SkeletalEstimator:
    def __init__(self, model_path=None, model_name="YOLO"):
        self.logger = setup_logger(__name__)
        self.model_path = model_path if model_path else config.model_weights_path
        self.model_name = model_name
        self.model = self.load_model()

    def load_model(self):
        if self.model_name == "YOLO":
            model = YOLO(self.model_path)
            self.logger.info(f"Loaded YOLO model from {self.model_path}")
        else:
            self.logger.error("Unsupported model name, only 'YOLO' is supported")
            raise ValueError("Unsupported model name, only 'YOLO' is supported")
        return model

    def GetBodyPoints(self, frame, mode="YOLO"):
        if mode == "YOLO":
            results = self.model(frame)
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            points = [(int(x), int(y)) if conf > 0.5 else (None, None) for x, y, conf in keypoints]
            probs = [conf for x, y, conf in keypoints]
        else:
            raise ValueError("Unsupported model name")
        return points, probs

    def FrameAddSkeleton(self, frame_, points_, MODE_="YOLO"):
        if MODE_ == "COCO":
            POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
        elif MODE_ == "MPI":
            POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
        elif MODE_ == "MOT16":
            POSE_PAIRS = [[2,3],[1,2],[1,5],[5,6],[3,4],[6,7],[1,8],[9,10],[10,11],[2,9],[5,12],[9,8],[8,12],[11,23],[11,24],[23,22],[24,22],[12,13],[13,14],[14,21],[21,20],[21,20],[21,19],[20,19],[1,0],[0,16],[0,15],[15,17],[16,18]]
        elif MODE_ == "YOLO":
            POSE_PAIRS = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            # インデックスが範囲内かを確認
            if partA < len(points_) and partB < len(points_):
                if points_[partA] and points_[partB] and None not in points_[partA] and None not in points_[partB]:
                    cv2.line(frame_, points_[partA], points_[partB], (0, 255, 255), 2)
                    cv2.circle(frame_, points_[partA], 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        return frame_

    def process_video(self, video_path, save_raw_frames=False, save_skeleton_frames=False):
        video_name = os.path.splitext(os.path.basename(video_path))[0]  # ファイル名のみを取得
        save_dir = os.path.join(config.output_data_path, video_name)
        frames_dir = os.path.join(save_dir, "raw_frames")
        skeleton_dir = os.path.join(save_dir, "skeleton_frames")

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(skeleton_dir, exist_ok=True)

        csv_landmarks_path = os.path.join(save_dir, f"{video_name}_landmarks.csv")
        csv_probs_path = os.path.join(save_dir, f"{video_name}_probs.csv")

        self.logger.info(f"Processing video {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open the video file: {video_path}")
            raise IOError("Cannot open the video file.")

        all_landmarks = []
        all_probs = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks, probs = self.estimate_skeleton(frame, frames_dir, skeleton_dir, frame_count, save_raw_frames, save_skeleton_frames)
            all_landmarks.append(landmarks)
            all_probs.append(probs)
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Video processing complete. Results saved in {save_dir}")

        # ランドマークリストをゼロパディング
        max_landmarks = max(len(l) for l in all_landmarks)
        all_landmarks_padded = [l + [(None, None)] * (max_landmarks - len(l)) for l in all_landmarks]

        # 信頼度リストをゼロパディング
        max_probs = max(len(p) for p in all_probs)
        all_probs_padded = [p + [0.0] * (max_probs - len(p)) for p in all_probs]

        # CSVに保存
        np.savetxt(csv_landmarks_path, np.array(all_landmarks_padded).reshape(len(all_landmarks_padded), -1), delimiter=",", fmt="%s")
        np.savetxt(csv_probs_path, np.array(all_probs_padded), delimiter=",", fmt="%s")

        return save_dir

    def estimate_skeleton(self, frame, frames_dir, skeleton_frames_dir, frame_count, save_raw_frames, save_skeleton_frames):
        points, probs = self.GetBodyPoints(frame, mode="YOLO")
        
        # raw_frames の画像を保存
        if save_raw_frames:
            raw_frame_filename = f"{frame_count:04d}.png"
            cv2.imwrite(os.path.join(frames_dir, raw_frame_filename), frame)

        # スケルトンを描画したフレームを生成
        if save_skeleton_frames:
            skeleton_frame = self.FrameAddSkeleton(frame.copy(), points, MODE_=self.model_name)
            skeleton_frame_filename = f"{frame_count:04d}.png"
            cv2.imwrite(os.path.join(skeleton_frames_dir, skeleton_frame_filename), skeleton_frame)

        return points, probs

if __name__ == "__main__":
    estimator = SkeletalEstimator()
    video_path = r"C:\Users\itohi\Downloads\archive\videos_160\videos_160\84.mp4"
    output_dir = estimator.process_video(video_path, save_raw_frames=True, save_skeleton_frames=True)
    print("Output saved in:", output_dir)
