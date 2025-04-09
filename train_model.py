import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM
from sklearn.metrics import confusion_matrix, classification_report

# 載入資料
X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
label_classes = np.load('data/label_encoder.npy')


cnn_lstm_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=False),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

cnn_lstm_model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

cnn_lstm_model.summary()

history_cnn_lstm = cnn_lstm_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)


cnn_lstm_model.save('data/models/cnn_lstm_model.h5')

def plot_confusion_matrix_dl(model, X, y, title):
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y, axis=1)
    cm = confusion_matrix(y_true, y_pred_classes)

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = [[f"{val * 100:.0f}%" for val in row] for row in cm_normalized]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized,
                annot=annot,
                fmt="",
                cmap='Blues',
                xticklabels=label_classes,
                yticklabels=label_classes,
                vmin=0.0, vmax=1.0
                )
    plt.title(f"Confusion Matrix for {title}")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

plot_confusion_matrix_dl(cnn_lstm_model, X_test, y_test, 'CNN + LSTM')

loss_cnn_lstm, acc_cnn_lstm = cnn_lstm_model.evaluate(X_test, y_test)
print(f"CNN + LSTM Test Accuracy: {acc_cnn_lstm:.4f}")

def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title(f'{title} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history_cnn_lstm, 'CNN + LSTM')
