from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Eğitilen modeli yükleyin
model = load_model('models/happiness_detector.h5')

def predict_happiness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    img = cv2.resize(img, (48, 48)) / 255.0
    img = img.reshape(1, 48, 48, 1)
    prediction = model.predict(img)
    happiness_percentage = prediction[0][1] * 100
    return happiness_percentage

@app.route('/esosolcer')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Görüntü yüklenmedi!"
    
    file = request.files['file']
    if file.filename == '':
        return "Görüntü seçilmedi!"
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        happiness_percentage = predict_happiness(file_path)
        os.remove(file_path)

        # Dinamik başlık ve renk ayarları
        if happiness_percentage > 40:
            title = "TEBRİKLER"
            title_color = "#dc3545"  # Kırmızı renk
        else:
            title = "Maalesef"
            title_color = "#dc3545"  # Kırmızı renk

        # Mutluluk yüzdesini metin formatında hazırlayın
        result_text = f"Şuan %{happiness_percentage:.2f} ESOŞ'sunuz!"
        return render_template(
            'result.html', 
            title=title, 
            title_color=title_color,
            result_text=result_text, 
            happiness_percentage=int(happiness_percentage)
        )

if __name__ == '__main__':
    app.run(debug=True)
