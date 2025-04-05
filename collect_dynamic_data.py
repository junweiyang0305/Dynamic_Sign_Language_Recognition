import cv2
import mediapipe as mp
import pandas as pd
import os
import time


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels = ['I_Love_You','How_Are_You','I_am_Fine_Thank_You']  

cap = cv2.VideoCapture(0)

sequence_length = 60 

with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(image_rgb)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('Hand Landmarks', image)

        if cv2.waitKey(10) & 0xFF == ord('s'):
            print("開始錄製，請做出手語動作...")
            sequence = []
            current_label = None

            frames_captured = 0 #新增

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.putText(image,
                    f"Recording: {frames_captured}/{sequence_length}",
                    (10, 30),  # 顯示位置
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Recording', image)


                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if results.left_hand_landmarks:
                    landmarks = results.left_hand_landmarks.landmark
                elif results.right_hand_landmarks:
                    landmarks = results.right_hand_landmarks.landmark
                else:
                    continue  # 若無手部關鍵點，跳過當前幀

                frame_data = []
                for lm in landmarks:
                    frame_data.extend([lm.x, lm.y, lm.z])
                sequence.append(frame_data)

                frames_captured += 1

                if len(sequence) == sequence_length:
                    break

            print("請輸入手語標籤：", labels)
            current_label = input("輸入標籤：")
            if current_label not in labels:
                print("無效的標籤，請重新錄製。")
            else:
                # 將序列儲存為CSV
                df = pd.DataFrame(sequence)
                df['Label'] = current_label
                filename = os.path.join(output_dir, f"Daytime_{current_label}_{int(time.time())}.csv")
                df.to_csv(filename, index=False)
                print(f"資料已儲存至 {filename}")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
