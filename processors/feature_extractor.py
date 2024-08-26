import numpy as np
import scipy.fftpack

class FeatureExtractor:
    def __init__(self):
        """
        特徴抽出を担当するクラス。異なる種類のデータセットから有用な特徴を抽出します。
        """
        pass

    def extract_features(self, data):
        """
        データから特徴量を抽出します。
        Args:
            data (np.ndarray): 特徴抽出を行うデータ。
        
        Returns:
            np.ndarray: 抽出された特徴量。
        """
        # 例：FFTを用いた周波数領域の特徴抽出
        features = scipy.fftpack.fft(data)
        # 実数部分の取得
        real_part = np.real(features)
        # パワースペクトラムの計算
        power_spectrum = np.abs(real_part) ** 2
        return power_spectrum

    def normalize_features(self, features):
        """
        特徴量の正規化を行います。
        Args:
            features (np.ndarray): 正規化する特徴量。
        
        Returns:
            np.ndarray: 正規化された特徴量。
        """
        return (features - np.mean(features)) / np.std(features)
