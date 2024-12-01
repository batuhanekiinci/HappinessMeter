import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Eğitilen modeli yükleyin
model = load_model('models/happiness_detector.h5')

# Görüntüde mutluluk yüzdesi tahmini yapan fonksiyon
def predict_happiness(image_path):
    # Görüntüyü yükleyin ve gri tonlamaya çevirin
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Geçersiz görüntü yolu!"
    img = cv2.resize(img, (48, 48)) / 255.0  # Yeniden boyutlandır ve normalize et
    img = img.reshape(1, 48, 48, 1)  # CNN için uygun formata getir

    # Model tahmini
    prediction = model.predict(img)
    happiness_percentage = prediction[0][1] * 100  # "Mutlu" sınıfının olasılığı
    return f"Mutluluk Yüzdesi: %{happiness_percentage:.2f} ESOŞ'sunuz!"

# Örnek bir görüntü için tahmin yap
image_path = "data/archive/test/happy/2_1.jpg"  # Tahmin yapmak istediğiniz görüntü yolu
result = predict_happiness(image_path)
print(result)
