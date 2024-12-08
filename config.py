import os
class Config:
    SECRET_KEY = os.urandom(24)
    Debug = True
    Host = '0.0.0.0'
    Port = 5000
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB maks ukuran file