class Config:
    def __init__(self):
        # モデルのトレーニング関連設定
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.001

        # データ関連の設定
        self.dataset_path = './data/'  # データファイルや動画を格納
        self.output_data_path = './output_data/'  # 推定結果の保存パス

        # モデル保存とログ関連の設定
        self.model_save_path = './models/'  # モデル定義ファイルを保存
        self.log_path = './logs/'  # ログを保存するディレクトリ

        # 骨格推定モデルの設定
        self.model_weights_path = './models/yolov8n-pose.pt'  # YOLOv8モデルのパス

        # フレーム処理の設定
        self.frame_width = 640  # 処理するフレームの幅
        self.frame_height = 480  # 処理するフレームの高さ

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

# プロジェクト全体で一つのConfigオブジェクトを使い回す
config = Config()
