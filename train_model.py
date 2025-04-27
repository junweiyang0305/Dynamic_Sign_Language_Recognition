# ============================================
# 1. 套件與資料
# ============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score)
from sklearn.manifold import TSNE
from tensorflow.keras import Model        # 取中間層
from pathlib import Path                  # 儲存模型用

# 讀取你前處理好的檔案
X_train = np.load('data/X_train.npy')
X_test  = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test  = np.load('data/y_test.npy')
label_classes = np.load('data/label_encoder.npy')   # e.g. array([...])

num_classes = y_train.shape[1]
time_steps, feat_dim = X_train.shape[1], X_train.shape[2]

# ============================================
# 2. 建立 1-D CNN + LSTM (Functional API)
# ============================================
inputs = tf.keras.Input(shape=(time_steps, feat_dim))

x = tf.keras.layers.Conv1D(64, 3, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.LSTM(100, name='lstm')(x)          # ← 命名很重要
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ============================================
# 3. 訓練
# ============================================
history = model.fit(
    X_train, y_train,
    epochs=35,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 另存成 .keras (官方建議的新版格式)
Path('data/models').mkdir(parents=True, exist_ok=True)
model.save('data/models/cnn_lstm_model.keras')

# ============================================
# 4-1. 四大分類指標
# ============================================
def evaluate_metrics(model, X, y, names):
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y,      axis=1)

    print('\n=== Classification Report ===')
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    macro_p  = precision_score(y_true, y_pred, average='macro')
    macro_r  = recall_score   (y_true, y_pred, average='macro')
    macro_f1 = f1_score       (y_true, y_pred, average='macro')
    acc      = (y_true == y_pred).mean()

    print(f"Macro-Precision : {macro_p:.4f}")
    print(f"Macro-Recall    : {macro_r:.4f}")
    print(f"Macro-F1-score  : {macro_f1:.4f}")
    print(f"Accuracy        : {acc:.4f}\n")

evaluate_metrics(model, X_test, y_test, label_classes)

# ============================================
# 4-2. 混淆矩陣（row-wise 百分比）
# ============================================
def plot_confusion_matrix(model, X, y, names, title):
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    y_true = np.argmax(y, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = [[f'{v*100:.0f}%' for v in row] for row in cm_pct]

    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                xticklabels=names, yticklabels=names,
                vmin=0., vmax=1., cbar_kws={'label':'Row-wise accuracy'},
                ax=ax, linewidths=.5)

    ax.set(title=title, xlabel='Predicted label', ylabel='True label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout(); plt.show()

plot_confusion_matrix(model, X_test, y_test, label_classes,
                      'Confusion Matrix for CNN + LSTM')

# ============================================
# 4-3. 學習曲線
# ============================================
def plot_history(h, title):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(h.history['accuracy'],     label='train acc')
    plt.plot(h.history['val_accuracy'], label='val acc')
    plt.title(f'{title} – Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(h.history['loss'],     label='train loss')
    plt.plot(h.history['val_loss'], label='val loss')
    plt.title(f'{title} – Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.show()

plot_history(history, 'CNN + LSTM')

# ============================================
# 4-4. t-SNE 2-D 視覺化（含耗時）
# ============================================
def plot_tsne(model, X, y, names, layer_name='lstm', perplexity=30):
    # 1) 取 LSTM 層輸出
    feat_model = Model(inputs=model.input,
                       outputs=model.get_layer(layer_name).output)
    emb = feat_model.predict(X, verbose=0)         # (N, 100)

    # 2) t-SNE
    import time
    start = time.time()
    print(f'[t-SNE] start, N = {emb.shape[0]}')
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(emb)
    print(f'[t-SNE] finished in {time.time()-start:.1f} s')

    # 3) 畫散佈圖
    plt.figure(figsize=(6,5))
    for i, lbl in enumerate(names):
        idx = np.argmax(y,1) == i
        plt.scatter(emb_2d[idx,0], emb_2d[idx,1], s=8, label=lbl)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title('t-SNE of LSTM embeddings')
    plt.xlabel('Dim-1'); plt.ylabel('Dim-2')
    plt.tight_layout(); plt.show()

plot_tsne(model, X_test, y_test, label_classes)

