import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

cnn_lstm_model = tf.keras.models.load_model('data/models/cnn_lstm_model_gs.h5')

# 載入標籤
label_classes = np.load('data/label_encoder.npy')

cap = cv2.VideoCapture(0)

sequence_length = 60  
capturing = False    
sequence = []        
frames_captured = 0   

predicted_msg = ""    # 用來儲存辨識結果, 以便在畫面上顯示


EXIT_AFTER_RECOGNITION = False

with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image_rgb)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            capturing = True
            frames_captured = 0
            sequence = []
            predicted_msg = ""  # 清空上次預測結果
            print("開始錄製 60 幀...")

        if capturing:
            if results.left_hand_landmarks:
                landmarks = results.left_hand_landmarks.landmark
            elif results.right_hand_landmarks:
                landmarks = results.right_hand_landmarks.landmark
            else:
                landmarks = None

            if landmarks:
                frame_data = []
                for lm in landmarks:
                    frame_data.extend([lm.x, lm.y, lm.z])
                sequence.append(frame_data)
                frames_captured += 1

            # 在畫面上顯示錄製進度
            cv2.putText(image, f"Recording: {frames_captured}/{sequence_length}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 當達到 60 幀 => 進行一次預測
            if frames_captured == sequence_length:
                capturing = False
                frames_captured = 0

                X_input = np.array(sequence).reshape(1, sequence_length, -1)

                cnn_lstm_pred = cnn_lstm_model.predict(X_input)

                cnn_lstm_class = label_classes[np.argmax(cnn_lstm_pred)]

                predicted_msg = (
                    #f"CNN => {cnn_class}\n"
                    f"Predict: {cnn_lstm_class}"
                )

                print("錄製完畢，辨識結果:")
                print(predicted_msg.replace("\n", "\n  "))

                if EXIT_AFTER_RECOGNITION:
                    break

        if predicted_msg != "":
            lines = predicted_msg.split('\n')
            y0 = 30
            for i, line in enumerate(lines):
                y = y0 + i * 30
                cv2.putText(image, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if key == ord('q'):
            break

        cv2.imshow('Realtime Dynamic Inference', image)

    cap.release()
    cv2.destroyAllWindows()
