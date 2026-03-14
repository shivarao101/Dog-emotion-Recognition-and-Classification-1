import pickle
import numpy as np
from scipy.io import wavfile
from mfcc_nw import melcepstrum
import tensorflow as tf
from tensorflow import keras
import glob
from sklearn.model_selection import train_test_split
with open('file1.pkl','rb') as file:
    melfreq=pickle.load(file)
mn=0
av=0
for i in range(len(melfreq)):
    mn+=len(melfreq[i][0])
av=mn/len(melfreq)
sz=20*np.ceil(av)
print(sz)
inter=[]
x=np.zeros(((int(sz)),115))
for i in range(len(melfreq)):
    re=np.reshape(melfreq[i],(len(melfreq[i])*len(melfreq[i][0]),))
    if len(re)> int(sz):
        re=re[0:int(sz)]
    else:
        re=np.array(list(re)+[0]*(int(sz)-len(re)))
        
    inter.append(re)
for i in range(np.size(x,1)):
    x[:,i]=inter[i]
xr=x.T
train_data=np.zeros((115,1,int(sz)))
for i in range(115):
          for j in range(1):
                    for k in range(int(sz)):
                              train_data[i][j][k]=xr[i][k]
train_data=np.array(train_data)
train_label=np.array([0]*36 + [1]*33 + [2]*23 + [3]*23)
train_label=train_label.T
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(1,int(sz))),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=50,validation_data=(X_test, y_test))
model.save('my_model1.keras')


