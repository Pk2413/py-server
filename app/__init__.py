from flask import Flask
from config import Config
from app.routes import main  
import cv2
import threading


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Daftarkan blueprint
    app.register_blueprint(main)
    return app