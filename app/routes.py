import os
from flask import Blueprint, request, jsonify, Response, render_template
import cv2
import numpy as np
import logging
import base64
import time

from methodknn import FeatureExtractor,PondClassifier
main = Blueprint('main', __name__)


    
@main.route("/ekstrak", methods=["POST"])
def eksekusi_code():
    # Mengambil file dari request
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Pastikan folder uploads ada
    os.makedirs("uploads", exist_ok=True)
    
    # Simpan file dengan nama asli di folder uploads
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Inisialisasi feature extractor dan classifier
    feature_extractor = FeatureExtractor()
    pond_classifier = PondClassifier(feature_extractor)
    
    # Crop dan resize gambar, simpan di folder yang sama dengan nama file asli
    processed_image = feature_extractor.load_image(file_path, file_path)
    
    # File data yang digunakan untuk training
    data_file = "Extraksi Fitur Kolam Lele.xlsx"
    
    # Training model
    pond_classifier.train(data_file)

    # Melakukan prediksi pada gambar yang diunggah
    try:
        predicted_category = pond_classifier.predict(file_path, output_file=data_file)
        accuracy = pond_classifier.accuracy
        logloss = pond_classifier.logloss
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Mengembalikan respons
    return jsonify({
        "predicted_category": predicted_category,
        "accuracy": accuracy,
        "log_loss": logloss,
        "processed_file": file.filename
    })
# Variabel global untuk menyimpan frame terakhir
latest_frame = None

@main.route('/upload', methods=['POST'])
def upload_image():
    global latest_frame
    try:
        # Proses upload gambar (gunakan kode sebelumnya)
        file_data = request.get_data()
        if file_data:
            nparr = np.frombuffer(file_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                latest_frame = frame
                return jsonify({'message': 'Image received successfully'}), 200
        
        return jsonify({'error': 'No valid image data'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/latest_frame')
def latest_frame_route():
    global latest_frame
    if latest_frame is not None:
        # Encode frame ke base64 untuk ditampilkan di web
        _, buffer = cv2.imencode('.jpg', latest_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': frame_base64})
    return jsonify({'error': 'No frame available'}), 404

@main.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                try:
                    # Encode gambar ke format JPEG
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    
                    if ret:
                        frame = buffer.tobytes()
                        
                        # Kirim frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Error generating frame: {e}")
    
    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@main.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'has_frame': latest_frame is not None
    })
    
@main.route('/')
def index():
    return render_template('index.html')