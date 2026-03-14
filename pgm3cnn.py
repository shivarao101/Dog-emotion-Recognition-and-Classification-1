import numpy as np
import glob
from scipy.io import wavfile
from mfcc_nw import melcepstrum
import tensorflow as tf
from tensorflow import keras
model1=keras.models.load_model('my_model1cnn')
path = glob.glob("E:/phd_proj_new/cat_test/*.wav")#Apple_Apple_scab
mfc=[]
for file in path:
    fs,a = wavfile.read(file)
    a=np.double(a)
    mfc1=melcepstrum(a,fs,512,256,20)
    mfc.append(mfc1)
path = glob.glob("E:/phd_proj_new/dog_bark_test/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    a=np.double(a)
    mfc1=melcepstrum(a,fs,512,256,20)
    mfc.append(mfc1)
path = glob.glob("E:/phd_proj_new/dog_grunt_test/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    a=np.double(a)
    mfc1=melcepstrum(a,fs,512,256,20)
    mfc.append(mfc1)
path = glob.glob("E:/phd_proj_new/dog_growl_test/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    a=np.double(a)
    mfc1=melcepstrum(a,fs,512,256,20)
    mfc.append(mfc1)
##########
mn=0
av=0
for i in range(len(mfc)):
    mn+=len(mfc[i][0])
av=mn/len(mfc)
#sz=20*np.ceil(av)
sz=8760
tol= sz - (94*93)
inter=[]
x=np.zeros((48,94,93))
for i in range(len(mfc)):
    re=np.reshape(mfc[i],(len(mfc[i])*len(mfc[i][0]),))
    if len(re)> int(sz)-tol:
        re=re[0:int(sz)-tol]
    else:
        re=np.array(list(re)+[0]*(int(sz)-tol-len(re)))
    re=re.reshape(94,93)   
    inter.append(re)
for i in range(48):
    x[i,:,:]=inter[i]
test_data=np.array(x)
test_label=np.array([0]*14 + [1]*13 + [2]*11 + [3]*10   )
test_label=test_label.T
print(model1.evaluate(test_data,test_label))
