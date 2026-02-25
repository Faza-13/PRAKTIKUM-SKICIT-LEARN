import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

DATA_DIR = r'D:\Kontrol Cerdas'  # pastikan ini folder dataset, bukan file .py

data = []
labels = []

ALLOWED_EXT = ('.jpg', '.jpeg', '.png')

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_path in os.listdir(dir_path):
        if not img_path.lower().endswith(ALLOWED_EXT):
            continue

        data_aux = []
        x_ = []
        y_ = []

        img_full_path = os.path.join(dir_path, img_path)
        img = cv2.imread(img_full_path)

        if img is None:
            print("Gagal baca file:", img_full_path)
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # normalisasi
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

with open(r'D:\Kontrol Cerdas\data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Selesai! Jumlah data:", len(data))
print("File pickle dibuat di: D:\\Kontrol Cerdas\\data.pickle")