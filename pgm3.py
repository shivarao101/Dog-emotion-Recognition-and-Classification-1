import numpy as np
import glob
from scipy import signal
from scipy.io import wavfile
from mfcc_nw import melcepstrum
import tensorflow as tf
from tensorflow import keras
wp=(2*3000)/16000
ws=(2*4000)/16000
ap=1
as1=20
N,wc=signal.buttord(wp,ws,ap,as1)
b1,a1=signal.butter(N,wc)
model1=tf.keras.models.load_model('my_model1lstm.keras')
path = glob.glob("E:/lms_filtering/cat_test_lmsfilt/*.wav")#Apple_Apple_scab
mfc=[]
for file in path:
    fs,a = wavfile.read(file)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    mfc.append(mfc1)
path = glob.glob("E:/lms_filtering/dog_bark_test_lmsfilt/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    if fs>20000:
        fs=fs/3
        a=signal.decimate(a,3)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    mfc.append(mfc1)
path = glob.glob("E:/lms_filtering/dog_grunt_test_lmsfilt/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    fs=fs/3
    a=signal.decimate(a,3)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    mfc.append(mfc1)
path = glob.glob("E:/lms_filtering/dog_growl_test_lmsfilt/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    if fs>20000:
        fs=fs/3
        a=signal.decimate(a,3)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    mfc.append(mfc1)
##########
mn=0
av=0
for i in range(len(mfc)):
    mn+=len(mfc[i][0])
av=mn/len(mfc)
#sz=20*np.ceil(av)
sz=5820
inter=[]
x=np.zeros(((int(sz)),48))
for i in range(len(mfc)):
    re=np.reshape(mfc[i],(len(mfc[i])*len(mfc[i][0]),))
    if len(re)> int(sz):
        re=re[0:int(sz)]
    else:
        re=np.array(list(re)+[0]*(int(sz)-len(re)))
        
    inter.append(re)
for i in range(np.size(x,1)):
    x[:,i]=inter[i]
xr=x.T
train_data=np.zeros((48,1,int(sz)))
for i in range(48):
          for j in range(1):
                    for k in range(int(sz)):
                              train_data[i][j][k]=xr[i][k]
test_data=np.array(train_data)
test_label=np.array([0]*14 + [1]*13 + [2]*11 + [3]*10 )
test_label=test_label.T
print(model1.evaluate(test_data,test_label))
# test_data1=np.zeros((48,4))
# test_data1[0:14,0]=1
# test_data1[14:27,1]=1
# test_data1[27:38,2]=1
# test_data1[38:,3]=1
ypred=model1.predict(test_data)
pred_cat=tf.argmax(ypred, axis=1)
print(pred_cat)
# cm=tf.math.confusion_matrix(pred_cat, test_label)
# print(cm)
