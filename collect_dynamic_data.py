import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

######################################
# (1) 動態補償函式 (量測亮度 + gamma)
######################################

def measure_brightness(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def decide_gamma(brightness, min_b=50, max_b=150):
    if brightness <= min_b:
        return 2.0
    elif brightness >= max_b:
        return 0.5
    else:
        ratio = (brightness - min_b) / (max_b - min_b)
        gamma = 2.0 - ratio * 1.0
        return gamma

######################################
# (2) 灰世界自動白平衡 (Gray World)
######################################

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

######################################
# (3) 增亮 (Gamma+CLAHE+WB)
######################################

def enhance_low_light(image, gamma=1.8, clipLimit=2.0, tileGridSize=(8,8)):
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0,i] = np.clip(pow(i/255.0, 1.0/gamma)*255.0, 0, 255)
    gamma_corrected = cv2.LUT(image, look_up_table)

    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l)
    merged_lab = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    balanced = auto_white_balance_gray_world(enhanced)
    return balanced

######################################
# (4) 去雜訊
######################################

def denoise_image(image_bgr, h=10, hColor=10, templateWindowSize=1, searchWindowSize=1):
    denoised = cv2.fastNlMeansDenoisingColored(
        src=image_bgr,
        dst=None,
        h=h,
        hColor=hColor,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize
    )
    return denoised

######################################
# (5) 資料收集主程式
######################################

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

labels = ['I_love_you','How_are_you','I_am_fine_thank_you',
          'How_old_are_you','What_your_name','You_look_so_young']
sequence_length = 60

default_output_dir = 'data'
if not os.path.exists(default_output_dir):
    os.makedirs(default_output_dir)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.7,
                          min_tracking_confidence=0.7) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 翻轉
        image = cv2.flip(frame, 1)

        # (A) 量測亮度 + 動態 gamma
        brightness = measure_brightness(image)
        auto_gamma = decide_gamma(brightness)

        # (B) 增亮
        enhanced_image = enhance_low_light(image, gamma=auto_gamma)

        # (C) 去雜訊
        denoised_image = denoise_image(enhanced_image, h=10, hColor=10)

        # (D) Mediapipe 偵測 (只是顯示用)
        image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        mp_drawing.draw_landmarks(denoised_image,
                                  results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(denoised_image,
                                  results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(denoised_image,
                    f"Brightness={brightness:.1f}, Gamma={auto_gamma:.2f}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Collect Data (AutoComp+Denoise)", denoised_image)

        key = cv2.waitKey(10) & 0xFF
        # 按 's' 鍵開始錄製
        if key == ord('s'):
            print("開始錄製...請做出手語動作")
            sequence = []
            frames_captured = 0

            while True:
                ret2, frame2 = cap.read()
                if not ret2:
                    break

                image2 = cv2.flip(frame2, 1)

                # 自動補償 + 去雜訊
                brightness2 = measure_brightness(image2)
                auto_gamma2 = decide_gamma(brightness2)
                enhanced2 = enhance_low_light(image2, gamma=auto_gamma2)
                denoised2 = denoise_image(enhanced2, h=10, hColor=10)

                # Mediapipe
                image2_rgb = cv2.cvtColor(denoised2, cv2.COLOR_BGR2RGB)
                results2 = holistic.process(image2_rgb)

                cv2.putText(denoised2,
                            f"Recording: {frames_captured}/{sequence_length}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                mp_drawing.draw_landmarks(denoised2,
                                          results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(denoised2,
                                          results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                cv2.imshow("Recording", denoised2)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                # 擷取關鍵點(右手優先)
                if results2.right_hand_landmarks:
                    landmarks = results2.right_hand_landmarks.landmark
                elif results2.left_hand_landmarks:
                    landmarks = results2.left_hand_landmarks.landmark
                else:
                    landmarks = None

                if landmarks:
                    frame_data = []
                    for lm in landmarks:
                        frame_data.extend([lm.x, lm.y, lm.z])
                    sequence.append(frame_data)
                    frames_captured += 1

                # 到達 60 幀 => 結束錄製
                if frames_captured == sequence_length:
                    break

            # 要求輸入標籤
            print("請輸入手語標籤：\n", labels)
            current_label = input("輸入標籤：")
            if current_label not in labels:
                print("無效標籤, 請重新錄製")
            else:
                df = pd.DataFrame(sequence)
                df['Label'] = current_label
                filename = os.path.join(default_output_dir,
                                        f"{current_label}_{int(time.time())}.csv")
                df.to_csv(filename, index=False)
                print(f"資料已儲存至 {filename}")

        # 按 'q' => 結束
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
