#!/usr/bin/env python3
# 空白鍵錄 60 幀 → 推論；畫面顯示 FPS / 推論耗時 / RAM / CPU / 溫度
import time, cv2, numpy as np, mediapipe as mp, tensorflow as tf, psutil, os, subprocess
from keras.initializers import Orthogonal
from picamera2 import Picamera2

MODEL_PATH = '/home/junwei/models/cnn_lstm_model.h5'
LABEL_PATH = '/home/junwei/models/label_encoder.npy'
SEQ_LEN    = 60

# ── 載入模型 ────────────────────────────────────────────────
labels = np.load(LABEL_PATH)
model  = tf.keras.models.load_model(MODEL_PATH,
           custom_objects={'Orthogonal': Orthogonal})
print('✅  model OK, classes =', len(labels))

# ── 開啟相機 ────────────────────────────────────────────────
cam = Picamera2()
cam.configure(cam.create_preview_configuration(
        main={'format':'BGR888', 'size':(640,480)}))
cam.start(); time.sleep(1)

# ── Mediapipe Hands ───────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils
DIM  = 63            # 21*3

# ── 系統資源 ───────────────────────────────────────────────
proc = psutil.Process(os.getpid())
def get_temp():
    try:
        out = subprocess.check_output(
              ['vcgencmd','measure_temp']).decode()
        return float(out.split('=')[1].split("'")[0])
    except Exception:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip())/1000

# ── 迴圈狀態變數 ───────────────────────────────────────────
capturing, seq = False, []
prediction = 'press <space>'
fps, infer_ms = 0, 0
t_prev, t_inf = time.time(), 0
ram_mb = cpu_pct = temp_c = 0
sys_tick = 0                   # 每 10 幀更新一次系統指標

try:
    while True:
        frm = cam.capture_array()
        frm = cv2.flip(frm,1)

        # ---------- Mediapipe ----------
        res = hands.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        hand_lm = res.multi_hand_landmarks[0] if res.multi_hand_landmarks else None
        if hand_lm:
            draw.draw_landmarks(frm, hand_lm, mp_hands.HAND_CONNECTIONS)

        # ---------- 錄影邏輯 ----------
        if capturing:
            feat = ([c for p in hand_lm.landmark for c in (p.x,p.y,p.z)]
                    if hand_lm else [0.0]*DIM)
            seq.append(feat)

            # 進度條
            cv2.rectangle(frm,(10,180),(10+int(200*len(seq)/SEQ_LEN),200),
                          (0,255,0),-1)
            cv2.rectangle(frm,(10,180),(210,200),(255,255,255),2)
            cv2.putText(frm,f'{len(seq):>2}/{SEQ_LEN}',(220,200),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            if len(seq)==SEQ_LEN:
                t_inf = time.time()
                x = np.asarray(seq,np.float32)[None,...]
                y = model(x,training=False).numpy()[0]
                infer_ms = (time.time()-t_inf)*1000
                prediction = labels[int(y.argmax())]
                seq.clear(); capturing=False

        # ---------- FPS ----------
        now = time.time()
        fps = 1/(now-t_prev); t_prev = now

        # ---------- 系統指標每 10 幀更新 ----------
        sys_tick += 1
        if sys_tick == 10:
            ram_mb  = proc.memory_info().rss/1048576
            cpu_pct = psutil.cpu_percent(interval=0)
            temp_c  = get_temp()
            sys_tick = 0

        # ---------- 疊字 ----------
        info = [
            f'FPS       : {fps:4.1f}',
            f'Infer (ms): {infer_ms:5.1f}',
            f'RAM (MB)  : {ram_mb:6.1f}',
            f'CPU (%)   : {cpu_pct:5.1f}',
            f'Temp (C)  : {temp_c:4.1f}',
            f'Pred      : {prediction}',
        ]
        for i,txt in enumerate(info,1):
            cv2.putText(frm,txt,(10,25*i),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.imshow('Sign-Language Realtime',frm)

        # ---------- 鍵盤 ----------
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'):
            break
        if k==ord(' '):
            capturing=True; seq.clear(); prediction='Recording...'

finally:
    hands.close(); cam.stop(); cv2.destroyAllWindows()
