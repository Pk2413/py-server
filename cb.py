import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import mysql.connector
import os
import time
import requests
from io import BytesIO
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

        H = hsv_image[:,:,0]
        S = hsv_image[:,:,1]
        V = hsv_image[:,:,2]

        meanH = np.mean(H)
        meanS = np.mean(S)
        meanV = np.mean(V)

        return [meanH, meanS, meanV]

    def download_image(self, image_url):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        return image

    def extract_features(self, image_np):
        color_features = self.extract_color_features(image_np)
        color_hsv = self.extract_color_hsv(image_np)
        features = color_features + color_hsv if color_hsv is not None else color_features
        return features

    def extract_features_and_save_single_image(self, image_path, output_file, knn_model):
        
        image = self.download_image(image_path)
        image_np = np.array(image)
        features = self.extract_features(image_np)
        features = np.array(features).reshape(1, -1)
        category = knn_model.predict(features)[0]

        columns = ['Warna R', 'Warna G', 'Warna B', 'Warna H', 'Warna S', 'Warna V', 'Kondisi']
        data = pd.DataFrame(np.column_stack([features, [category]]), columns=columns)

        if os.path.exists(output_file):
            existing_data = pd.read_excel(output_file)
            data = pd.concat([existing_data, data], ignore_index=True)

        data.to_excel(output_file, index=False)

        print(f"Prediksi Kategori: {category}")

class PlantsClassifier:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.accuracy = 0
        self.db_connection = None

    def train(self, data_file):

        data = pd.read_excel(data_file)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.knn_classifier.fit(X_train, y_train)

        y_pred = self.knn_classifier.predict(X_test)

        conf_matrix = confusion_matrix(y_test, y_pred)

        TP = conf_matrix[1, 1]
        TN = conf_matrix[0, 0]
        FP = conf_matrix[0, 1]
        FN = conf_matrix[1, 0]

        self.accuracy = ((TP + TN) / (TP + FP + FN + TN)) * 100

    def simpan(self, category, model_accuracy):
        cursor = None
        url = 'https://greensense.kencang.id/greensense/insert.php'
        
        try:
            
            data_to_send = {
                'kategori': category,
                'akurasi': model_accuracy
            }

           
            response = requests.post(url, data=data_to_send)

            
            if response.status_code == 200:
                print("Data berhasil disimpan ke database.")
            else:
                print(f"Gagal menyimpan data. Kode status: {response.status_code}")

        except Exception as e:
            print(f"Error while saving to database: {e}")

        finally:
            if cursor is not None:
                cursor.close()

if __name__ == "__main__":
    feature_extractor = FeatureExtractor()
    plants_classifier = PlantsClassifier(feature_extractor)

    data_file = "https://greensense.kencang.id/greensense/ExtrasifiturGreenSense.xlsx"
    plants_classifier.train(data_file)

    output_file = "https://greensense.kencang.id/greensense/ExtrasifiturGreenSense.xlsx"

    try:
        while True:
            sample_image_path = 'https://greensense.kencang.id/greensense/CekKondisi/cekgambar.jpg'

            image = feature_extractor.download_image(sample_image_path)
            image_np = np.array(image)

            feature_extractor.extract_features_and_save_single_image(sample_image_path, output_file, plants_classifier.knn_classifier)

            new_image_features = feature_extractor.extract_features(image_np)

            predicted_category = plants_classifier.knn_classifier.predict([new_image_features])[0]
            model_accuracy = plants_classifier.accuracy

            plants_classifier.simpan(predicted_category, model_accuracy)

            print(f"Akurasi data test: {model_accuracy:.2f}%")

            time.sleep(30)

    except KeyboardInterrupt:
        print("Process interrupted. Exiting.")