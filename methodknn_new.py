import cv2
import numpy as np
import pandas as pd
from PIL import Image  # Untuk manipulasi gambar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def extract_color_features(self, image_np):
        """Ekstraksi rata-rata nilai RGB dari gambar."""
        r, g, b = cv2.split(image_np)
        avg_r = np.mean(r)
        avg_g = np.mean(g)
        avg_b = np.mean(b)
        return [avg_r, avg_g, avg_b]

    def extract_color_hsv(self, image_np):
        """Ekstraksi rata-rata nilai HSV dari gambar."""
        hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        H, S, V = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
        return [np.mean(H), np.mean(S), np.mean(V)]

    def load_image(self, image_path):
        """Memuat gambar, cropping, dan mengubah ukuran ke 224x224."""
        image = Image.open(image_path)
        # Mendapatkan ukuran gambar
        width, height = image.size
        # Menentukan ukuran crop (contoh: 100x100)
        crop_width, crop_height = 100, 100
        # Koordinat crop yang terpusat
        left = (width - crop_width) // 2
        upper = (height - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height
        # Crop dan ubah ukuran gambar
        cropped_image = image.crop((left, upper, right, lower))
        resized_image = cropped_image.resize((224, 224), Image.Resampling.LANCZOS)

        return np.array(resized_image)

    def extract_features(self, image_path):
        """Gabungan fitur RGB dan HSV."""
        image_np = self.load_image(image_path)
        color_features = self.extract_color_features(image_np)
        color_hsv = self.extract_color_hsv(image_np)
        return color_features + color_hsv


class PondClassifier:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Gunakan KNN dengan k=3
        self.scaler = StandardScaler()  # Normalisasi data
        self.accuracy = 0

    def train(self, data_file):
        """Melatih model KNN menggunakan data dari file Excel."""
        data = pd.read_excel(data_file)

        # Pisahkan fitur (X) dan label (y)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Normalisasi fitur menggunakan StandardScaler
        X = self.scaler.fit_transform(X)

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Latih model KNN
        self.knn_classifier.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = self.knn_classifier.predict(X_test)
        prob_preds = self.knn_classifier.predict_proba(X_test)

        # Hitung metrik evaluasi
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        self.accuracy = (conf_matrix.diagonal().sum() / conf_matrix.sum()) * 100
        logloss = log_loss(y_test, prob_preds)

        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{report}")
        print(f"Akurasi Model: {self.accuracy:.2f}%")
        print(f"Log Loss: {logloss:.4f}")

    def predict(self, image_path):
        """Prediksi kategori untuk gambar baru."""
        features = self.feature_extractor.extract_features(image_path)
        features = np.array(features).reshape(1, -1)

        # Normalisasi fitur baru
        features = self.scaler.transform(features)

        # Prediksi kategori
        predicted_category = self.knn_classifier.predict(features)[0]
        print(f"Prediksi Kategori: {predicted_category}")
        return predicted_category


if __name__ == "__main__":
    # Inisialisasi
    feature_extractor = FeatureExtractor()
    Pond_Classifier = PondClassifier(feature_extractor)

    # File data untuk pelatihan
    data_file = r".\Extraksi Fitur Kolam Lele.xlsx"

    # Pelatihan model
    Pond_Classifier.train(data_file)

    # Prediksi untuk gambar baru
    sample_image_path = r".\foto\th (2).jpg"
    Pond_Classifier.predict(sample_image_path)