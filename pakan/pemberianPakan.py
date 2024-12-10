import numpy as np
import pandas as pd
import logging
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Nonaktifkan warning
warnings.filterwarnings('ignore', category=RuntimeWarning)

class Predictor:
    def __init__(self):
        # Inisialisasi logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Inisialisasi atribut dataset
        self.dataset_pakan = None
        self.dataset_servo = None
        
        # Path dataset
        dataset_pakan_path = './pakan/dataset_pakan.xlsx'
        dataset_servo_path = './pakan/dataset_servo.xlsx'
        
        # Load dataset
        self.dataset_pakan = self._load_dataset(dataset_pakan_path)
        self.dataset_servo = self._load_dataset(dataset_servo_path)

        # Inisialisasi model
        self.model_pakan = self._create_pakan_model()
        self.model_servo = self._create_servo_model()

    def _load_dataset(self, dataset_path):
        """Load dataset dari Excel"""
        try:
            # Baca file Excel
            df = pd.read_excel(dataset_path)
            self.logger.info(f"Dataset berhasil dimuat: {dataset_path}")
            return df
        except Exception as e:
            self.logger.error(f"Gagal memuat dataset: {e}")
            return None

    def _create_pakan_model(self):
        """Buat model prediksi pakan"""
        try:
            if self.dataset_pakan is None:
                raise ValueError("Dataset pakan tidak tersedia")

            # Validasi kolom
            required_columns = ['umur_ikan', 'pakan_per_ekor']
            if not all(col in self.dataset_pakan.columns for col in required_columns):
                raise ValueError(f"Dataset pakan harus memiliki kolom: {required_columns}")

            # Persiapkan data
            X = self.dataset_pakan['umur_ikan'].values.reshape(-1, 1)
            y = self.dataset_pakan['pakan_per_ekor'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Gunakan pipeline dengan polynomial features
            model = make_pipeline(
                PolynomialFeatures(degree=2),
                LinearRegression()
            )
            
            # Fit model
            model.fit(X_train, y_train)

            # Evaluasi model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            self.logger.info(f"Model Pakan - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

            return model
        except Exception as e:
            self.logger.error(f"Gagal membuat model pakan: {e}")
            return None

    def _create_servo_model(self):
        """Buat model prediksi servo"""
        try:
            if self.dataset_servo is None:
                raise ValueError("Dataset servo tidak tersedia")

            # Validasi kolom
            required_columns = ['total_pakan', 'waktu_bukaan']
            if not all(col in self.dataset_servo.columns for col in required_columns):
                raise ValueError(f"Dataset servo harus memiliki kolom: {required_columns}")

            # Persiapkan data
            X = self.dataset_servo['total_pakan'].values.reshape(-1, 1)
            y = self.dataset_servo['waktu_bukaan'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Gunakan pipeline dengan polynomial features
            model = make_pipeline(
                PolynomialFeatures(degree=2),
                LinearRegression()
            )
            
            # Fit model
            model.fit(X_train, y_train)

            # Evaluasi model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            self.logger.info(f"Model Servo - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

            return model
        except Exception as e:
            self.logger.error(f"Gagal membuat model servo: {e}")
            return None

    def predict_pakan(self, umur_ikan, jumlah_ikan):
        """Prediksi jumlah pakan"""
        try:
            # Validasi input
            umur_ikan = float(umur_ikan)
            jumlah_ikan = float(jumlah_ikan)

            # Validasi model
            if self.model_pakan is None:
                raise ValueError("Model pakan belum diinisialisasi")

            # Persiapkan input untuk prediksi
            X = np.array([[umur_ikan]])

            # Prediksi pakan per ekor
            jumlah_pakan_per_ekor = self.model_pakan.predict(X)[0]
            
            # Total pakan
            total_pakan = jumlah_pakan_per_ekor * jumlah_ikan
            
            # Logging
            self.logger.info(f"Prediksi Pakan - Umur: {umur_ikan}, Jumlah Ikan: {jumlah_ikan}")
            self.logger.info(f"Pakan per Ekor: {jumlah_pakan_per_ekor:.2f}")
            self.logger.info(f"Total Pakan: {total_pakan:.2f}")
            
            return max(0, total_pakan)
        except Exception as e:
            self.logger.error(f"Error prediksi pakan: {e}")
            return 0

    def predict_servo(self, total_pakan):
        """Prediksi waktu bukaan servo"""
        try:
            # Validasi input
            total_pakan = float(total_pakan)

            # Validasi model
            if self.model_servo is None:
                raise ValueError("Model servo belum diinisialisasi")

            # Persiapkan input untuk prediksi
            X = np.array([[total_pakan]])

            # Prediksi waktu bukaan
            waktu_bukaan = self.model_servo.predict(X)[0]
            
            # Logging
            self.logger.info(f"Prediksi Servo - Total Pakan: {total_pakan}")
            self.logger.info(f"Waktu Bukaan: {waktu_bukaan:.2f}")
            
            return max(0, waktu_bukaan)
        except Exception as e:
            self.logger.error(f"Error prediksi servo: {e}")
            return 0
        
        
