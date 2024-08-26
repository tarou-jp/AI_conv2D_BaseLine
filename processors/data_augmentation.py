import numpy as np
import cv2

class DataAugmentation:
    def __init__(self):
        """
        データ拡張を担当するクラス。異なるデータ拡張技術を適用して、データセットの多様性を高めます。
        """
        pass

    def rotate_image(self, image, angle):
        """
        画像を指定された角度で回転させます。
        Args:
            image (np.ndarray): 回転させる画像。
            angle (float): 回転角度。
        
        Returns:
            np.ndarray: 回転された画像。
        """
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
        return rotated_image

    def flip_image(self, image, flip_code):
        """
        画像を水平または垂直に反転します。
        Args:
            image (np.ndarray): 反転させる画像。
            flip_code (int): 0は垂直反転、1は水平反転、-1は両方。
        
        Returns:
            np.ndarray: 反転された画像。
        """
        return cv2.flip(image, flip_code)

    def add_noise(self, image, noise_type='gaussian', mean=0, var=0.01):
        """
        画像にノイズを加えます。
        Args:
            image (np.ndarray): ノイズを加える画像。
            noise_type (str): ノイズの種類 ('gaussian', 'salt_pepper', etc.)
            mean (float): ガウシアンノイズの平均。
            var (float): ガウシアンノイズの分散。
        
        Returns:
            np.ndarray: ノイズが加えられた画像。
        """
        row, col, ch = image.shape
        if noise_type == 'gaussian':
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_pepper':
            s_vs_p = 0.5
            amount = 0.004
            noisy_image = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            noisy_image[coords[0], coords[1], :] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            noisy_image[coords[0], coords[1], :] = 0
        return noisy_image
