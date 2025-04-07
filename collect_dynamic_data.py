import cv2
import mediapipe as mp
import pandas as pd
import os
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

default_output_dir = 'data'
if not os.path.exists(default_output_dir):
    os.makedirs(default_output_dir)

labels = ['I_Love_You','How_Are_You','I_am_Fine_Thank_You']  

folder_choices = {
    '1': 'data',       # 亮度正常的資料
    '2': 'dark_data'   # 低光環境的資料
}

cap = cv2.VideoCapture(0)

sequence_length = 60 

with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
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
            frames_captured = 0 

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.putText(
                    image,
                    f"Recording: {frames_captured}/{sequence_length}",
                    (10, 30), 
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
                # 讓使用者選擇要存放的資料夾
                print("選擇要存放的資料夾：")
                for key, folder in folder_choices.items():
                    print(f"{key}. {folder}")
                folder_input = input("請輸入選項 (預設存 'data')：")

                if folder_input in folder_choices:
                    chosen_output_dir = folder_choices[folder_input]
                else:
                    chosen_output_dir = default_output_dir 

                if not os.path.exists(chosen_output_dir):
                    os.makedirs(chosen_output_dir)

                df = pd.DataFrame(sequence)
                df['Label'] = current_label
                filename = os.path.join(
                    chosen_output_dir,
                    f"Daytime_{current_label}_{int(time.time())}.csv"
                )
                df.to_csv(filename, index=False)
                print(f"資料已儲存至 {filename}")

        # 按下 'q' 鍵退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
