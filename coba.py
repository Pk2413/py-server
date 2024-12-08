import cv2
import numpy as np
import os
import pandas as pd

class ImageProcessing:
    def extract_color_features(self, image_np):
        r, g, b = cv2.split(image_np)
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        return [avg_r, avg_g, avg_b]

    def extract_color_hsv(self, image_np):
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        H, S, V = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
        return [np.mean(H), np.mean(S), np.mean(V)]

    def process_images_in_folder(self, folder_path, excel_file):
        data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Tambahkan format gambar yang diinginkan
                image_path = os.path.join(folder_path, filename)
                image_np = cv2.imread(image_path)

                rgb_features = self.extract_color_features(image_np)
                hsv_features = self.extract_color_hsv(image_np)

                data.append([filename, *rgb_features, *hsv_features])

        # Membuat DataFrame dari data yang dikumpulkan
        df = pd.DataFrame(data, columns=['Filename', 'Avg_R', 'Avg_G', 'Avg_B', 'Avg_H', 'Avg_S', 'Avg_V'])

        # Menyimpan ke file Excel, jika file sudah ada, tambahkan ke bawahnya
        if os.path.exists(excel_file):
            df.to_excel(excel_file, index=False, header=False, startrow=pd.read_excel(excel_file).shape[0] + 1)
        else:
            df.to_excel(excel_file, index=False)

# Contoh penggunaan
folder_path = 'uploads'  # Ganti dengan path folder Anda
excel_file = 'output_colors.xlsx'  # Nama file Excel yang diinginkan

color_extractor = ImageProcessing()
color_extractor.process_images_in_folder(folder_path, excel_file)