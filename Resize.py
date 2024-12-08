import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                img = img.resize(size, Image.Resampling.LANCZOS)
                output_path = os.path.join(output_folder, filename)
                img.save(output_path)
                print(f"Resized image saved to {output_path}")

# Folder berisi gambar asli
input_folder = r'C:\Users\septi\Downloads\foto air kolam testing'

# Folder tempat menyimpan gambar hasil resize
output_folder = r'C:\Users\septi\Downloads\output test'

# Ukuran baru untuk gambar (lebar, tinggi)
size = (224, 224)

resize_images(input_folder, output_folder, size)