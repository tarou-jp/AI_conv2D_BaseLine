from models.layers.attention_layers import CustomAttention
from models.layers.convolutional_layers import CustomConv2D
from models.layers.recurrent_layers import CustomLSTM
import tensorflow as tf
from utils.config import config

class InferenceEngine:
    def __init__(self, model_path=None):
        # モデルのパスが指定されていない場合、デフォルトのパスを使用
        if model_path is None:
            model_path = config.model_save_path + "best_model.h5"

        # モデルをロード
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        # モデルを指定されたパスからロード
        try:
            model = tf.keras.models.load_model(model_path, custom_objects={
                'CustomConv2D': CustomConv2D,
                'CustomLSTM': CustomLSTM,
                'CustomAttention': CustomAttention
            })
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            raise

    def predict(self, input_data):
        # 推論を実行
        predictions = self.model.predict(input_data, batch_size=config.batch_size)
        return predictions
