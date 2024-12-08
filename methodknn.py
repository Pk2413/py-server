import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss, classification_report
import os
import time
from PIL import Image


class FeatureExtractor:
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


    def load_image(self, file_path, save_path=None):
        # Buka gambar
        image = Image.open(file_path)
        
        # Mendapatkan ukuran gambar
        width, height = image.size
        
        # Menentukan ukuran crop (misalnya 100x100)
        crop_width, crop_height = 100, 100
        
        # Menghitung koordinat crop yang terpusat
        left = (width - crop_width) // 2
        upper = (height - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height
        
        # Melakukan cropping
        cropped_image = image.crop((left, upper, right, lower))
        
        # Ubah ukuran gambar hasil cropping
        resized_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Simpan gambar yang sudah diproses jika save_path disediakan
        if save_path:
            resized_image.save(save_path)
        
        return np.array(resized_image)

    def extract_features(self, image_np):
        # Ekstraksi fitur RGB dan HSV
        color_features = self.extract_color_features(image_np)
        color_hsv = self.extract_color_hsv(image_np)
        return color_features + color_hsv


class PondClassifier:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.knn_classifier = KNeighborsClassifier(n_neighbors=3)
        self.accuracy = 0
        self.logloss =0

    def train(self, data_file):
        # Membaca data dari file Excel
        data = pd.read_excel(data_file)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Training model KNN
        self.knn_classifier.fit(X_train, y_train)

        # Prediksi dan menghitung metrik evaluasi
        y_pred = self.knn_classifier.predict(X_test)
        prob_preds = self.knn_classifier.predict_proba(X_test)

        
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        self.accuracy = (conf_matrix.diagonal().sum() / conf_matrix.sum()) * 100
        self.logloss = log_loss(y_test, prob_preds)

        # Menampilkan metrik
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{report}")
        print(f"Akurasi model: {self.accuracy:.2f}%")
        print(f"Log Loss: {self.logloss:.4f}")

    def predict(self, image_path, output_file):
        # Prediksi kategori untuk gambar baru
        image_np = self.feature_extractor.load_image(image_path)
        features = self.feature_extractor.extract_features(image_np)
        
        # Check extracted features
        print(f"Extracted features: {features}")
        
        predicted_category = self.knn_classifier.predict([features])[0]
        print(f"Raw predicted category: {predicted_category}")

        # Simpan hasil prediksi ke file
        columns = ['Warna R', 'Warna G', 'Warna B', 'Warna H', 'Warna S', 'Warna V', 'Kondisi']
        data = pd.DataFrame([features + [predicted_category]], columns=columns)

        if os.path.exists(output_file):
            existing_data = pd.read_excel(output_file)
            data = pd.concat([existing_data, data], ignore_index=True)

        data.to_excel(output_file, index=False)
        print(f"Prediksi kategori: {predicted_category}")

        # Mengubah nilai predicted_category menjadi string yang sesuai
        if predicted_category == 1:
            predicted_category = "keruh"
        elif predicted_category == 2:  # Assuming 0 is the other category
            predicted_category = "jernih"
        else:
            predicted_category = None  # Handle unexpected values
        
        return predicted_category


# if __name__ == "__main__":
#     feature_extractor = FeatureExtractor()
#     pond_classifier = PondClassifier(feature_extractor)

#     data_file = "C:/dataset/EkstraksiFitur/Extraksi Fitur Kolam Lele.xlsx"
#     output_file = "C:/dataset/EkstraksiFitur/Extraksi Fitur Kolam Lele.xlsx"
#     sample_image_path = "C:/dataset/EkstraksiFitur/foto/th (2).jpg"

#     # Training model dan menampilkan hasil evaluasi
#     pond_classifier.train(data_file)

#     # Prediksi untuk gambar baru
#     try:
#         while True:
#             pond_classifier.predict(sample_image_path, output_file)
#             time.sleep(5)
#     except KeyboardInterrupt:
#         print("Proses dihentikan oleh pengguna. Keluar.")
#     except Exception as e:
#         print(f"Terjadi kesalahan: {e}")
