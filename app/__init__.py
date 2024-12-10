from flask import Flask
from flask_cors import CORS
from config import Config
from app.routes import main
import ssl
import os

import firebase_admin
from firebase_admin import credentials, db

from pakan.pemberianPakan import Predictor

def create_app():
    app = Flask(__name__)
    
    app.predictor = Predictor()
    
        # Initialize Firebase Admin SDK
    # Dapatkan path absolut ke direktori app
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Konstruksi path lengkap ke service account key
    SERVICE_ACCOUNT_PATH = os.path.join(
        BASE_DIR, 
        'aq-farm-firebase-adminsdk-upvwf-28d96b930c.json'
    )

    # Inisialisasi Firebase
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://aq-farm-default-rtdb.asia-southeast1.firebasedatabase.app'
        })
        print("Firebase berhasil diinisialisasi")
    except Exception as e:
        print(f"Gagal inisialisasi Firebase: {e}")
    # Aktifkan CORS untuk semua rute
    CORS(app, resources={
        r"/*": {
            "origins": "*",  # Sesuaikan dengan kebutuhan
            "allow_headers": [
                "Content-Type", 
                "Authorization", 
                "Access-Control-Allow-Methods",
                "Access-Control-Allow-Origin"
            ],
            "supports_credentials": True
        }
    })

    app.config.from_object(Config)

    # Daftarkan blueprint
    app.register_blueprint(main)
    
    return app

# Untuk SSL (opsional)
def create_ssl_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('path/to/cert.pem', 'path/to/key.pem')
    return context

# Modifikasi di main.py atau app.py
# if __name__ == '_main_':
#     app, _ = create_app()
    
#     # Pilih salah satu:
#     # 1. Tanpa SSL
#     app.run(host='0.0.0.0', port=5000, debug=True)
    
    # 2. Dengan SSL
    # ssl_context = create_ssl_context()
    # app.run(host='0.0.0.0', port=5000, ssl_context=ssl_context, debug=True