import glob
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.io import wavfile
from mfcc_nw import melcepstrum
import pickle
def tran(a):
    r,c=np.shape(a)
    ls=np.zeros((r))
    for i in range(r):
        ls[i]=sum(abs(a[i,:]))
    return ls
wp=(2*3000)/16000
ws=(2*4000)/16000
ap=1
as1=20
N,wc=signal.buttord(wp,ws,ap,as1)
b1,a1=signal.butter(N,wc)
c1cnt=0
c2cnt=0
c3cnt=0
c4cnt=0
path = glob.glob("E:/lms_filtering/cat_train_lmsfilt/*.wav")#Apple_Apple_scab
mfc=[]
for file in path:
    fs,a = wavfile.read(file)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    #mfc2=tran(mfc1)
    mfc.append(mfc1)
path = glob.glob("E:/lms_filtering/dog_bark_train_lmsfilt/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    if fs>20000:
        c2cnt+=1
        fs=fs/3
        a=signal.decimate(a,3)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    #mfc2=tran(mfc1)
    mfc.append(mfc1)
path = glob.glob("E:/lms_filtering/dog_grunt_train_lmsfilt/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    if fs>20000:
        c3cnt+=1
    fs=fs/3
    a=signal.decimate(a,3)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    #mfc2=tran(mfc1)
    mfc.append(mfc1)
#mfc=np.array(mfc)
#print(len(mfc))
path = glob.glob("E:/lms_filtering/dog_growl_train_lmsfilt/*.wav")#Apple_Apple_scab
for file in path:
    fs,a = wavfile.read(file)
    if fs>20000:
        c4cnt+=1
        fs=fs/3
        a=signal.decimate(a,3)
    a=np.double(a)
    y=signal.lfilter(b1,a1,a)
    mfc1=melcepstrum(y,fs,512,256,20)
    mfc.append(mfc1)
with open('file1.pkl','wb') as file:
    pickle.dump(mfc,file)
