import cv2
import numpy as np

def process_camera_frame(image_path):
    """
    Fungsi untuk memproses frame kamera.
    Contoh: deteksi wajah, pengolahan gambar, dll.
    """
    try:
        # Baca gambar
        image = cv2.imread(image_path)
        
        # Contoh: deteksi wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return {
            'total_faces_detected': len(faces),
            'faces_coordinates': faces.tolist()
        }
    except Exception as e:
        return {
            'error': str(e)
        }