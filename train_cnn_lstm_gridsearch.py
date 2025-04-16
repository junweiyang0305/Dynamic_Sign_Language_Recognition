############################################################
# train_cnn_lstm_gridsearch.py
# 使用 CNN+LSTM 進行動態手語識別之 Grid Search + K-Fold + EarlyStopping
############################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


import os

# 1. 載入資料
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
label_classes = np.load('data/label_encoder.npy')  # 若需要最終測試後繪製混淆矩陣的標籤


############################################################
# 2. 建立 create_cnn_lstm_model 函式
############################################################
def create_cnn_lstm_model(kernel_size=3,
                          filters=64,
                          lstm_units=100,
                          dropout_rate=0.5,
                          learning_rate=0.001,
                          input_shape=(60, 63),  # (time_steps, features)
                          num_classes=3):
    """
    根據傳入的參數建構 CNN+LSTM 模型並 compile。
    """
    model = Sequential()
    # Conv1D Layer 1
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Conv1D Layer 2
    model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM Layer
    model.add(LSTM(lstm_units, return_sequences=False))

    # Dense + Dropout
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


############################################################
# 3. 定義要搜尋的參數空間
############################################################

kernel_sizes = [3, 5]
filters_list = [64, 128]
lstm_units_list = [100, 128]
learning_rates = [0.001, 0.0005]
dropouts = [0.3, 0.5]
batch_sizes = [16, 32]
epochs = 30  # 可以固定，也可做候選列表

############################################################
# 4. 交叉驗證 (K-Fold) + Grid Search + EarlyStopping
############################################################

k_folds = 3  # k-fold 數量
best_acc = -1
best_config = None

print("開始執行 Grid Search + K-Fold 交叉驗證...")

for ks in kernel_sizes:
    for f in filters_list:
        for lstm_u in lstm_units_list:
            for lr in learning_rates:
                for dr in dropouts:
                    for bs in batch_sizes:
                        # 先儲存 k-fold 的驗證集分數
                        fold_accuracies = []

                        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                        fold_index = 1

                        for train_idx, val_idx in kf.split(X_train):
                            X_tr, X_val = X_train[train_idx], X_train[val_idx]
                            y_tr, y_val = y_train[train_idx], y_train[val_idx]

                            # 建立模型
                            model = create_cnn_lstm_model(
                                kernel_size=ks,
                                filters=f,
                                lstm_units=lstm_u,
                                dropout_rate=dr,
                                learning_rate=lr,
                                input_shape=(X_train.shape[1], X_train.shape[2]),
                                num_classes=y_train.shape[1]
                            )

                            # EarlyStopping
                            early_stop = EarlyStopping(
                                monitor='val_loss',
                                patience=3,
                                restore_best_weights=True
                            )

                            # 訓練
                            history = model.fit(
                                X_tr, y_tr,
                                epochs=epochs,
                                batch_size=bs,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stop],
                                verbose=0
                            )

                            # 驗證集評估
                            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                            fold_accuracies.append(val_acc)

                            print(f"[Fold {fold_index}] ks={ks}, filters={f}, lstm={lstm_u}, lr={lr}, drop={dr}, "
                                  f"bs={bs}, val_acc={val_acc:.4f}")
                            fold_index += 1

                        mean_acc = np.mean(fold_accuracies)

                        # 若平均驗證準確率高於目前最佳，更新最佳參數
                        if mean_acc > best_acc:
                            best_acc = mean_acc
                            best_config = (ks, f, lstm_u, lr, dr, bs)

print("Grid Search 結束！")
print(f"最佳平均驗證準確率: {best_acc:.4f}")
print(f"最佳組合參數: kernel_size={best_config[0]}, filters={best_config[1]}, "
      f"lstm_units={best_config[2]}, learning_rate={best_config[3]}, dropout={best_config[4]}, "
      f"batch_size={best_config[5]}")

############################################################
# 5. 以最佳參數組合做最終訓練，並在測試集上檢驗
############################################################

ks_best, f_best, lstm_best, lr_best, dr_best, bs_best = best_config

print("\n以最佳參數組合做最終訓練...")
final_model = create_cnn_lstm_model(
    kernel_size=ks_best,
    filters=f_best,
    lstm_units=lstm_best,
    dropout_rate=dr_best,
    learning_rate=lr_best,
    input_shape=(X_train.shape[1], X_train.shape[2]),
    num_classes=y_train.shape[1]
)

early_stop_final = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 最終訓練：可用全部 X_train, y_train，也可切出部分當 validation
history_final = final_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=bs_best,
    validation_split=0.1,  # 或直接使用 X_test 做 validation
    callbacks=[early_stop_final],
    verbose=1
)

# 在測試集上評估
loss_final, acc_final = final_model.evaluate(X_test, y_test, verbose=1)
print(f"最終測試集準確率: {acc_final:.4f}")

############################################################
# 6. (選擇性) 繪製混淆矩陣 + 保存最終模型
############################################################

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(model, X, y, title="Final Model"):
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = [[f"{val * 100:.0f}%" for val in row] for row in cm_normalized]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=annot, fmt="", cmap='Blues',
                xticklabels=label_classes,
                yticklabels=label_classes,
                vmin=0.0, vmax=1.0)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


plot_confusion_matrix(final_model, X_test, y_test, title="CNN+LSTM (Best Config)")

# 保存最終模型
if not os.path.exists("data/models"):
    os.makedirs("data/models")
final_model.save("data/models/cnn_lstm_model_gs.h5")
print("最終 CNN+LSTM 模型已儲存於 data/models/cnn_lstm_model_gs.h5")
