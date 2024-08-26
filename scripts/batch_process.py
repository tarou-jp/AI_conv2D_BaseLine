import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.skelton_estimator import SkeletalEstimator

def process_videos_from_directory(directory_path, csv_path):
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)

    # SkeletalEstimatorの初期化
    estimator = SkeletalEstimator()

    # CSVの各行に対して処理を実行
    for index, row in df.iterrows():
        video_id = row['id']
        slow_motion = row['slow']  # スローモーションかどうかを確認

        # スローモーションの場合はスキップ
        if slow_motion == 1:
            print(f"Skipping video {video_id} because it is in slow motion.")
            continue

        video_name = f"{video_id}.mp4"  # ディレクトリ内の動画ファイル名を生成
        video_path = os.path.join(directory_path, video_name)

        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist.")
            continue

        # 動画を処理して結果を保存（トリミングなし）
        output_dir = estimator.process_video(video_path)
        print(f"Processed {video_name} and saved results in {output_dir}")

if __name__ == "__main__":
    directory_path = r"C:\Users\itohi\Downloads\archive\videos_160\videos_160"  # 動画ディレクトリのパスを指定
    csv_path = r"C:\Users\itohi\Downloads\archive\GolfDB.csv"          # CSVファイルのパスを指定

    process_videos_from_directory(directory_path, csv_path)
