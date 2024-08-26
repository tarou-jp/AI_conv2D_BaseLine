import cv2

class VideoProcessor:
    def __init__(self, resize_width=224, resize_height=224):
        """
        動画データの前処理を担当するクラス。
        Args:
            resize_width (int): リサイズ後のフレームの幅。
            resize_height (int): リサイズ後のフレームの高さ。
        """
        self.resize_width = resize_width
        self.resize_height = resize_height

    def process_video(self, video_path):
        """
        動画ファイルを処理してフレームをリサイズし、リストとして返す。
        Args:
            video_path (str): 処理する動画ファイルのパス。
        
        Returns:
            list of ndarray: リサイズされたフレームのリスト。
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            frames.append(resized_frame)
        cap.release()
        return frames

    def display_frame(self, frame, window_name='Frame'):
        """
        単一のフレームをウィンドウで表示する。
        Args:
            frame (ndarray): 表示するフレーム。
            window_name (str): 表示ウィンドウの名前。
        """
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)  # キーが押されるまで待つ
        cv2.destroyAllWindows()
