import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# (1) (亮度測量 + Gamma )

def measure_brightness(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def decide_gamma(brightness, min_b=50, max_b=150):
    if brightness <= min_b:
        return 2.5  # 極暗 => 強力增亮
    elif brightness >= max_b:
        return 1.0  # 已足夠亮 => 不增亮
    else:
        ratio = (brightness - min_b) / (max_b - min_b)
        gamma = 2.5 - ratio * 1.5  # 在 1.0~2.5 之間
        return gamma

# (2) 灰階白平衡

def auto_white_balance_gray_world(image_bgr):
    b, g, r = cv2.split(image_bgr)
    b_float = b.astype(np.float32)
    g_float = g.astype(np.float32)
    r_float = r.astype(np.float32)

    mean_b = np.mean(b_float)
    mean_g = np.mean(g_float)
    mean_r = np.mean(r_float)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    if mean_b == 0: mean_b = 1e-6
    if mean_g == 0: mean_g = 1e-6
    if mean_r == 0: mean_r = 1e-6

    kb = mean_gray / mean_b
    kg = mean_gray / mean_g
    kr = mean_gray / mean_r

    b_corr = np.clip(b_float * kb, 0, 255).astype(np.uint8)
    g_corr = np.clip(g_float * kg, 0, 255).astype(np.uint8)
    r_corr = np.clip(r_float * kr, 0, 255).astype(np.uint8)

    return cv2.merge((b_corr, g_corr, r_corr))

# (3) 增亮: Gamma + CLAHE + 白平衡

def enhance_low_light(image, gamma=1.8, clipLimit=2.0, tileGridSize=(8,8)):
    # A. Gamma Correction
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0,i] = np.clip(pow(i/255.0, 1.0/gamma)*255.0, 0, 255)
    gamma_corrected = cv2.LUT(image, look_up_table)

    # B. LAB + CLAHE
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)
    merged_lab = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # C. 灰世界自動白平衡
    balanced_result = auto_white_balance_gray_world(enhanced)

    return balanced_result

# (4) 去雜訊函式

def denoise_image(image_bgr, h=10, hColor=10, templateWindowSize=5, searchWindowSize=6):
    denoised = cv2.fastNlMeansDenoisingColored(
        src=image_bgr,
        dst=None,
        h=h,
        hColor=hColor,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize
    )
    return denoised

# (5) Main: 即時推論程式

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 載入 CNN+LSTM
cnn_lstm_model = tf.keras.models.load_model('data/models/cnn_lstm_model.h5')
label_classes = np.load('data/label_encoder.npy')

cap = cv2.VideoCapture(0)
sequence_length = 60
capturing = False
sequence = []
frames_captured = 0
predicted_msg = ""
EXIT_AFTER_RECOGNITION = False

with mp_holistic.Holistic(min_detection_confidence=0.6,
                          min_tracking_confidence=0.6) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 翻轉
        image = cv2.flip(frame, 1)

        # 1. 量測亮度 => 動態 gamma
        brightness = measure_brightness(image)
        auto_gamma = decide_gamma(brightness)

        # 2. 增亮
        enhanced_image = enhance_low_light(image, gamma=auto_gamma)

        # 3. 去雜訊
        denoised_image = denoise_image(enhanced_image, h=10, hColor=10)

        # 4. Mediapipe
        image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # 繪製關鍵點
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(denoised_image, results.left_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(denoised_image, results.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS)

        # 錄製邏輯 (按空白鍵 => 開始錄 60 幀)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            capturing = True
            frames_captured = 0
            sequence = []
            predicted_msg = ""
            print("開始錄製 60 幀...")

        if capturing:
            landmarks = None
            if results.right_hand_landmarks:
                landmarks = results.right_hand_landmarks.landmark
            elif results.left_hand_landmarks:
                landmarks = results.left_hand_landmarks.landmark

            if landmarks:
                frame_data = []
                for lm in landmarks:
                    frame_data.extend([lm.x, lm.y, lm.z])
                sequence.append(frame_data)
                frames_captured += 1

            # 顯示錄製進度
            cv2.putText(denoised_image, f"Recording: {frames_captured}/{sequence_length}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if frames_captured == sequence_length:
                capturing = False
                frames_captured = 0

                X_input = np.array(sequence).reshape(1, sequence_length, -1)
                cnn_lstm_pred = cnn_lstm_model.predict(X_input)
                cnn_lstm_class = label_classes[np.argmax(cnn_lstm_pred)]
                predicted_msg = f"Predict: {cnn_lstm_class}"
                print("錄製完畢，辨識結果:")
                print("  " + predicted_msg)

                if EXIT_AFTER_RECOGNITION:
                    break

        # 顯示推論結果
        if predicted_msg != "":
            lines = predicted_msg.split('\n')
            y0 = 60
            for i, line in enumerate(lines):
                y = y0 + i*30
                cv2.putText(denoised_image, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        # 在畫面上順便顯示亮度 / gamma
        text_info = f"Brightness={brightness:.1f}, Gamma={auto_gamma:.2f}"
        cv2.putText(denoised_image, text_info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if key == ord('q'):
            break

        cv2.imshow('Realtime Inference (AutoComp + Denoise)', denoised_image)

cap.release()
cv2.destroyAllWindows()