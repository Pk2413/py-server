import os
from flask import Blueprint, request, jsonify, Response, render_template
# from flask_socketio import emit
import cv2
import numpy as np
import logging
import base64
import time
from datetime import datetime, timezone, date

from firebase_admin import db

from methodknn import FeatureExtractor,PondClassifier
from pakan.pemberianPakan import Predictor

predictor = Predictor()
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

    hasil = db.reference('pcv/hasilFoto')
    hasil.set(predicted_category)
    
    # Mengembalikan respons
    return jsonify({
        "predicted_category": predicted_category,
        "accuracy": accuracy,
        "log_loss": logloss,
        "processed_file": file.filename
    })
# Variabel global untuk menyimpan frame terakhir
latest_frame = None

# Fungsi untuk mendapatkan frame terbaru (bisa dari kamera atau sumber lain)
def get_latest_frame():
    global latest_frame
    # Contoh: Simulasi dengan mengambil gambar dari webcam
    cap = cv2.VideoCapture(0)  # Menggunakan webcam untuk contoh
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        latest_frame = frame  # Menyimpan frame terbaru
    else:
        print("Gagal mengambil frame")

# Endpoint untuk menerima gambar dari client
@main.route('/upload', methods=['POST'])
def upload_image():
    global latest_frame
    try:
        # Ambil data gambar dari request
        file_data = request.get_data()
        if file_data:
            nparr = np.frombuffer(file_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                latest_frame = frame  # Update frame terbaru
                return jsonify({'message': 'Image received successfully'}), 200
        
        return jsonify({'error': 'No valid image data'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk streaming gambar terbaru

@main.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        frame_count = 0
        while frame_count < 500:  # Batasi jumlah frame
            if latest_frame is not None:
                try:
                    # Encode gambar ke format JPEG
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    
                    if ret:
                        frame = buffer.tobytes()
                        
                        # Kirim frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        
                        frame_count += 1
                        time.sleep(0.1)  # Kontrol rate frame
                except Exception as e:
                    print(f"Error generating frame: {e}")
                    break
            else:
                time.sleep(0.5)  # Tunggu frame
    
    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Tambahkan route khusus untuk mendapatkan frame terakhir
@main.route('/latest_frame')
def get_latest_frame():
    global latest_frame
    if latest_frame is not None:
        # Encode frame ke base64
        _, buffer = cv2.imencode('.jpg', latest_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': frame_base64})
    return jsonify({'error': 'No frame available'}), 404
# Status endpoint untuk memeriksa apakah frame tersedia
@main.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'has_frame': latest_frame is not None
    })
    
@main.route('/')
def index():
    return render_template('index.html')



# Inisialisasi Predictor dengan path dataset
predictor = Predictor()

@main.route('/pakan', methods=['GET'])
def predict_pakan():
    try:
        # Log untuk debugging
        logging.info("Memulai prediksi pakan")

        # Ambil data dari Firebase
        try:
            # Reference ke path data di Firebase
            ref_jumlah = db.reference('/data/jumlahIkan')
            jumlah_ikan = ref_jumlah.get()

            ref_tanggal_mulai = db.reference('/informasiLele/tanggalMulai')
            tanggal_mulai = ref_tanggal_mulai.get()
        except Exception as firebase_error:
            logging.error(f"Error mengambil data dari Firebase: {firebase_error}")
            return jsonify({
                'status': 'error',
                'message': 'Gagal mengambil data dari Firebase',
                'error': str(firebase_error)
            }), 404

        # Validasi data
        if jumlah_ikan is None or tanggal_mulai is None:
            logging.warning("Data ikan tidak lengkap")
            return jsonify({
                'status': 'error',
                'message': 'Data ikan tidak ditemukan'
            }), 404

        # Konversi jumlah ikan ke numerik
        try:
            jumlah_ikan = int(jumlah_ikan)
        except (ValueError, TypeError) as convert_error:
            logging.error(f"Gagal konversi jumlah ikan: {convert_error}")
            return jsonify({
                'status': 'error',
                'message': 'Jumlah ikan harus berupa angka',
                'error': str(convert_error)
            }), 400

        # Hitung usia ikan
        try:
            # Normalisasi format tanggal
            if tanggal_mulai.endswith('Z'):
                tanggal_mulai = tanggal_mulai[:-1]
            
            # Parse tanggal dengan metode yang lebih robust
            try:
                # Pertama coba ISO format
                tanggal_awal = datetime.fromisoformat(tanggal_mulai)
            except ValueError:
                try:
                    # Jika gagal, coba parsing manual
                    tanggal_awal = datetime.strptime(tanggal_mulai, "%Y-%m-%d")
                except ValueError:
                    # Jika masih gagal, lempar error
                    raise ValueError(f"Format tanggal tidak valid: {tanggal_mulai}")
            
            # Tambahkan timezone jika tidak ada
            if tanggal_awal.tzinfo is None:
                tanggal_awal = tanggal_awal.replace(tzinfo=timezone.utc)
            
            # Gunakan datetime.now() dengan timezone yang sama
            tanggal_sekarang = datetime.now(timezone.utc)
            
            # Hitung usia, pastikan tidak negatif
            umur_ikan = max(0, (tanggal_sekarang - tanggal_awal).days)
            
            # Debug logging
            logging.info(f"Tanggal Mulai: {tanggal_awal}")
            logging.info(f"Tanggal Sekarang: {tanggal_sekarang}")
            logging.info(f"Umur Ikan: {umur_ikan} hari")

        except Exception as date_error:
            logging.error(f"Gagal menghitung usia ikan: {date_error}")
            return jsonify({
                'status': 'error',
                'message': 'Gagal menghitung usia ikan',
                'error': str(date_error),
                'tanggal_input': tanggal_mulai
            }), 400

        # Prediksi pakan
        total_pakan = predictor.predict_pakan(umur_ikan, jumlah_ikan)
        waktu_bukaan = predictor.predict_servo(total_pakan)

        # Persiapkan response JSON
        response = {
            'status': 'success',
            'data': {
                'jumlah_ikan': jumlah_ikan,
                'tanggal_mulai': tanggal_mulai,
                'umur_ikan_hari': umur_ikan,
                'total_pakan': round(total_pakan, 2),
                'waktu_bukaan': int(waktu_bukaan)
            }
        }

        # Log hasil
        logging.info(f"Prediksi berhasil - Pakan: {total_pakan}, Waktu Bukaan: {waktu_bukaan}")

        return jsonify(response), 200

    except Exception as e:
        # Tangkap semua error yang mungkin terjadi
        logging.error(f"Error umum dalam prediksi pakan: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Terjadi kesalahan internal',
            'error': str(e)
        }), 500
        
        
@main.route('/setWaktuMakan')
def set_waktu_makan():
    try:
        # Ambil tanggal mulai dari /informasiLele/tanggalMulai
        tanggal_mulai_ref = db.reference('informasiLele/tanggalMulai')
        tanggal_mulai_str = tanggal_mulai_ref.get()
        
        if not tanggal_mulai_str:
            return jsonify({
                'status': 'error',
                'message': 'Tanggal mulai tidak ditemukan'
            }), 404

        # Konversi string tanggal ke datetime
        # Gunakan parsing yang sesuai dengan format ISO
        tanggal_mulai = datetime.fromisoformat(tanggal_mulai_str.replace('Z', '+00:00')).date()
        
        # Hitung usia ikan
        usia_hari = (date.today() - tanggal_mulai).days

        # Tentukan status makan berdasarkan usia
        if 0 <= usia_hari <= 10:
            status_makan_pagi = 'ON'
            status_makan_sore = 'ON'
            status_makan_malam = 'OFF'
        elif 11 <= usia_hari <= 31:
            status_makan_pagi = 'ON'
            status_makan_sore = 'ON'
            status_makan_malam = 'ON'
        else:  # > 31
            status_makan_pagi = 'ON'
            status_makan_sore = 'ON'
            status_makan_malam = 'OFF'

        # Update status makan di database
        makan_pagi_ref = db.reference('makanPagi/status')
        makan_pagi_ref.set(status_makan_pagi)
        makan_sore_ref = db.reference('makanSore/status')
        makan_sore_ref.set(status_makan_sore)
        
        makan_malam_ref = db.reference('makanMalam/status')
        makan_malam_ref.set(status_makan_malam)

        return jsonify({
            'status': 'success',
            'message': 'Status makan berhasil diatur',
            'tanggal_mulai': tanggal_mulai.isoformat(),
            'usia_hari': usia_hari,
            'status_makan_pagi': status_makan_pagi
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500