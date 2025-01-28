import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
import time

# 初始化 Mediapipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 載入機器學習模型 (pkl)
rf_model = joblib.load('data/models/RandomForest_model.pkl')
svm_model = joblib.load('data/models/SVM_model.pkl')
knn_model = joblib.load('data/models/KNN_model.pkl')

# 載入深度學習模型 (h5)
cnn_model = tf.keras.models.load_model('data/models/cnn_model.h5')
cnn_lstm_model = tf.keras.models.load_model('data/models/cnn_lstm_model.h5')

# 載入標籤
label_classes = np.load('data/label_encoder.npy')

# 初始化攝影機
cap = cv2.VideoCapture(0)

sequence_length = 60  # 設定一次錄製的幀數
capturing = False     # 是否正在錄製
sequence = []         # 儲存當前錄製的關鍵點序列
frames_captured = 0   # 已錄製的幀數

predicted_msg = ""    # 用來儲存辨識結果, 以便在畫面上顯示

# 如果您要「一次錄完就退出程式」，可以把下方這個 flag 改成 True
EXIT_AFTER_RECOGNITION = False

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 翻轉與色彩轉換
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe 偵測
        results = holistic.process(image_rgb)

        # 繪製手部關鍵點
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 取得鍵盤輸入 (每幀執行一次)
        key = cv2.waitKey(1) & 0xFF

        # 1) 按下空白鍵 => 開始錄製
        if key == ord(' '):
            capturing = True
            frames_captured = 0
            sequence = []
            predicted_msg = ""  # 清空上次預測結果, 以免持續顯示
            print("開始錄製 60 幀...")

        # 2) 若正在錄製 => 收集關鍵點
        if capturing:
            # 擷取 landmarks
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

                # 準備資料形狀
                X_input = np.array(sequence).reshape(1, sequence_length, -1)

                # 機器學習模型預測
                rf_pred = rf_model.predict(X_input.reshape(1, -1))
                svm_pred = svm_model.predict(X_input.reshape(1, -1))
                knn_pred = knn_model.predict(X_input.reshape(1, -1))

                # 深度學習模型預測
                cnn_pred = cnn_model.predict(X_input)
                cnn_lstm_pred = cnn_lstm_model.predict(X_input)

                # 取得預測類別
                rf_class = label_classes[rf_pred[0]]
                svm_class = label_classes[svm_pred[0]]
                knn_class = label_classes[knn_pred[0]]
                cnn_class = label_classes[np.argmax(cnn_pred)]
                cnn_lstm_class = label_classes[np.argmax(cnn_lstm_pred)]

                # 組合成字串, 用以在畫面顯示
                predicted_msg = (
                    f"RandomForest => {rf_class}\n"
                    f"SVM => {svm_class}\n"
                    f"KNN => {knn_class}\n"
                    f"CNN => {cnn_class}\n"
                    f"CNN+LSTM => {cnn_lstm_class}"
                )

                print("錄製完畢，辨識結果:")
                print(predicted_msg.replace("\n", "\n  "))  # 印在終端機

                # 如果您要一次錄完就退出程式 => break
                if EXIT_AFTER_RECOGNITION:
                    break

        # 3) 在畫面上顯示 predicted_msg
        #    若 predicted_msg 非空, 顯示多行文字
        if predicted_msg != "":
            lines = predicted_msg.split('\n')
            y0 = 70  # 從哪裡開始印
            for i, line in enumerate(lines):
                y = y0 + i * 30
                cv2.putText(image, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 4) 按下 'q' => 離開
        if key == ord('q'):
            break

        cv2.imshow('Realtime Dynamic Inference', image)

    cap.release()
    cv2.destroyAllWindows()
