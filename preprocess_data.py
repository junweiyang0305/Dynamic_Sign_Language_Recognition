import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

file_paths = glob.glob('data/*.csv')
data_frames = [pd.read_csv(fp) for fp in file_paths]
data = pd.concat(data_frames, ignore_index=True)

X = data.drop(columns=['Label']).values
y = data['Label'].values

sequence_length = 60  

def create_sequences(X, y, seq_length):
    sequences = []
    labels = []
    for i in range(len(X) - seq_length + 1):
        seq = X[i:i+seq_length]
        label = y[i+seq_length-1]  
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(X, y, sequence_length)

le = LabelEncoder()
y_encoded = le.fit_transform(y_seq)
y_onehot = to_categorical(y_encoded)

# 分割訓練集、測試集
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
)

# 儲存處理後的資料
if not os.path.exists('data/models'):
    os.makedirs('data/models')

np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)
np.save('data/label_encoder.npy', le.classes_)

print("資料前處理完成並儲存。")
