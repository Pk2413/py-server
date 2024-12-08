import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. Baca dataset dari file CSV
data = pd.read_csv('C:/dataset/EkstraksiFitur/Extraksi Fitur Kolam Lele.csv')

# 2. Ekstrak fitur RGB dan HSV serta label kondisi
rgb_hsv_features = data[['Warna R', 'Warna G', 'Warna B', 'Warna H', 'Warna S', 'Warna V']].values
labels = data['Kondisi'].values

# 3. Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(rgb_hsv_features, labels, test_size=0.3, random_state=42)

# 4. Normalisasi data untuk meningkatkan performa KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Inisialisasi model KNN dan latih model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 6. Prediksi dan evaluasi model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Menghitung probabilitas prediksi untuk menghitung log_loss
y_pred_proba = knn.predict_proba(X_test)

# Menghitung log loss
loss = log_loss(y_test, y_pred_proba)

# Menampilkan akurasi
print("Accuracy:", accuracy)

# Menampilkan log loss
print("Log Loss:", loss)

# Menghitung dan menampilkan matriks konfusi
print("\nConfusion Matrix:")
print(conf_matrix)

# Menampilkan laporan klasifikasi (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Keruh', 'Jernih']))

# 7. Tampilkan Confusion Matrix dalam bentuk heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Keruh', 'Jernih'], yticklabels=['Keruh', 'Jernih'])
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix of KNN Model")
# plt.show()

# 8. Fungsi untuk memprediksi kondisi air dari gambar
def predict_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224))  # Sesuaikan ukuran sesuai kebutuhan
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(resized_image, axis=0)
    avg_rgb = np.average(avg_color_per_row, axis=0)
    avg_hsv = np.average(np.average(hsv_image, axis=0), axis=0)
    features = np.hstack((avg_rgb, avg_hsv)).reshape(1, -1)
    features = scaler.transform(features)
    prediction = knn.predict(features)
    return prediction[0]

# 9. Contoh prediksi dari gambar input
image_path = 'C:/dataset/EkstraksiFitur/foto/th (2).jpg'  # Ganti dengan path gambar Anda
prediksi_kondisi = predict_image(image_path)
print("Prediksi kondisi air:", prediksi_kondisi)