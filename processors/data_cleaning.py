import numpy as np

class DataCleansing:
    def __init__(self):
        """
        データクリーニングを行うためのクラス。
        このクラスは、欠損値の処理やノイズの除去、外れ値の検出といったタスクを担当します。
        """
        pass

    def remove_outliers(self, data, threshold=3.0):
        """
        Zスコアを使用して外れ値を除去します。
        Args:
            data (np.ndarray): 外れ値を検出するデータ配列。
            threshold (float): 外れ値と見なすZスコアの閾値。
        
        Returns:
            np.ndarray: 外れ値が除去されたデータ配列。
        """
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return data[z_scores < threshold]

    def fill_missing_values(self, data, strategy='mean'):
        """
        欠損値を指定された戦略に従って埋めます。
        Args:
            data (np.ndarray): 欠損値を埋めるデータ配列。
            strategy (str): 'mean', 'median', 'zero' のうち、使用する戦略。
        
        Returns:
            np.ndarray: 欠損値が埋められたデータ配列。
        """
        if strategy == 'mean':
            fill_value = np.mean(data[~np.isnan(data)])
        elif strategy == 'median':
            fill_value = np.median(data[~np.isnan(data)])
        elif strategy == 'zero':
            fill_value = 0
        else:
            raise ValueError("Unsupported fill strategy.")

        filled_data = np.where(np.isnan(data), fill_value, data)
        return filled_data

    def remove_noise(self, data, filter_strength=1.0):
        """
        シンプルな平滑化を通じてデータからノイズを除去します。
        Args:
            data (np.ndarray): ノイズ除去を行うデータ配列。
            filter_strength (float): 平滑化の強度。
        
        Returns:
            np.ndarray: ノイズが除去されたデータ配列。
        """
        return np.convolve(data, np.ones((filter_strength,))/filter_strength, mode='valid')
