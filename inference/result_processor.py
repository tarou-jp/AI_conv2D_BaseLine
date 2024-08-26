import numpy as np

class ResultProcessor:
    def __init__(self):
        pass

    def process(self, predictions):
        # 推論結果の後処理を行うメソッド
        processed_results = self.decode_predictions(predictions)
        return processed_results

    def decode_predictions(self, predictions):
        # 予測結果を解釈可能な形式に変換
        decoded = np.argmax(predictions, axis=-1)
        return decoded
