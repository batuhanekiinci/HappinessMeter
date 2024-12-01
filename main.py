import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# 1. Veri Yükleme
def load_data(data_dir):
    X = []
    y = []

    # Etiket Haritalama: Klasör adlarını etiketlerle eşleştir
    label_mapping = {
        "happy": 1,       # Mutlu sınıfı
        "angry": 0,       # Mutlu Değil
        "sad": 0,         # Mutlu Değil
        "surprised": 0,   # Mutlu Değil
        "neutral": 0,     # Mutlu Değil
    }

    for label in os.listdir(data_dir):  # Her etiket klasörünü dolaş
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):  # Klasör olup olmadığını kontrol et
            if label not in label_mapping:
                continue  # Bilinmeyen bir etiket varsa atla
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamaya çevir
                if img is None:  # Geçersiz görüntüleri atla
                    continue
                img = cv2.resize(img, (48, 48))  # Görüntüyü yeniden boyutlandır
                X.append(img)
                y.append(label_mapping[label])  # Haritalanmış etiket kullan
    return np.array(X), np.array(y)


# Eğitim ve Test Verilerini Yükleme
train_dir = "data/archive/train"
test_dir = "data/archive/test"

X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)

# Veriyi Normalize Etme
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Etiketleri one-hot encode etme
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 2. Modeli Oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # İki sınıf: Mutlu ve Mutlu Değil
])

# Modeli Derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Modeli Eğitme
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 4. Modeli Kaydetme
model.save('models/happiness_detector.h5')
print("Model başarıyla kaydedildi!")
