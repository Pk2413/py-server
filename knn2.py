import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import os
import time

# Fungsi untuk membaca dataset dan melatih model
def train_model(data_file):
    # Baca dataset dari file Excel
    data = pd.read_excel(data_file)
    rgb_hsv_features = data[['Warna R', 'Warna G', 'Warna B', 'Warna H', 'Warna S', 'Warna V']].values
    labels = data['Kondisi'].values

    # Split data menjadi data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(rgb_hsv_features, labels, test_size=0.3, random_state=42)

    # Normalisasi data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Inisialisasi model KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Evaluasi model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    y_pred_proba = knn.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_proba)

    # Menampilkan hasil evaluasi
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Log Loss: {loss:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Keruh', 'Jernih']))

    return knn, scaler

# Fungsi untuk memprediksi gambar dan memperbarui dataset
def predict_and_update(image_path, knn, scaler, data_file):
    # Ekstraksi fitur gambar
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(resized_image, axis=0)
    avg_rgb = np.average(avg_color_per_row, axis=0)
    avg_hsv = np.average(np.average(hsv_image, axis=0), axis=0)
    features = np.hstack((avg_rgb, avg_hsv)).reshape(1, -1)

    # Normalisasi fitur
    features = scaler.transform(features)

    # Prediksi kategori
    prediction = knn.predict(features)[0]
    print(f"Prediksi kondisi air: {prediction}")

    # Perbarui dataset dengan prediksi baru
    new_data = pd.DataFrame([np.append(features.flatten(), prediction)], columns=['Warna R', 'Warna G', 'Warna B', 'Warna H', 'Warna S', 'Warna V', 'Kondisi'])
    if os.path.exists(data_file):
        existing_data = pd.read_excel(data_file)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    # Simpan dataset yang diperbarui ke file Excel
    updated_data.to_excel(data_file, index=False)

if __name__ == "__main__":
    # Path ke dataset dan gambar
    data_file = "C:/dataset/EkstraksiFitur/Extraksi Fitur Kolam Lele.xlsx"
    image_path = "C:/dataset/EkstraksiFitur/outputresize/jernih/Image_2 (1).jpeg"

    # Latih model
    knn, scaler = train_model(data_file)

    # Iterasi prediksi setiap 24 jam
    try:
        while True:
            predict_and_update(image_path, knn, scaler, data_file)

            # Tunggu 24 jam sebelum iterasi berikutnya
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Proses dihentikan oleh pengguna. Keluar.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
