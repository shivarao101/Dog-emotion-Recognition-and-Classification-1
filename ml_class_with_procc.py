import pickle
import numpy as np
from mfcc_nw import melcepstrum
import tensorflow as tf
from tensorflow import keras
import glob
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier,BernoulliRBM
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from scipy import signal
from scipy.io import wavfile
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
with open('file1.pkl','rb') as file:
    melfreq=pickle.load(file)
mn=0
av=0
for i in range(len(melfreq)):
    mn+=len(melfreq[i][0])
av=mn/len(melfreq)
sz=20*np.ceil(av)
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
train_data=xr
# train_data=np.zeros((115,1,int(sz)))
# for i in range(115):
#           for j in range(1):
#                     for k in range(int(sz)):
#                               train_data[i][j][k]=xr[i][k]
train_data=np.array(train_data)
train_label=np.array([0]*36 + [1]*33 + [2]*23 + [3]*23)
train_label=train_label.T
#clf=GradientBoostingClassifier(n_estimators=50, learning_rate=0.5,max_depth=2, random_state=0)
clf=MLPClassifier(alpha=0.1, max_iter=1000, random_state=42)
#clf = LinearDiscriminantAnalysis()
#clf=LogisticRegression(solver='newton-cg',max_iter=1000)
#clf=KNeighborsClassifier(5)
#clf=tree.DecisionTreeClassifier(max_depth=2)
#clf=RandomForestClassifier(max_depth=2,criterion="gini")
#clf=SVC(kernel='rbf',degree=3)
clf.fit(train_data,train_label)
print(clf.score(train_data,train_label))

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(1,int(sz))),
#     keras.layers.Dense(256, activation='relu'),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(4, activation='softmax')
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_data, train_label, epochs=50)
# model.save('my_model1')
wp=(2*3000)/16000
ws=(2*4000)/16000
ap=1
as1=20
N,wc=signal.buttord(wp,ws,ap,as1)
b1,a1=signal.butter(N,wc)
#model1=keras.models.load_model('my_model1')
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
test_data=xr
# train_data=np.zeros((48,1,int(sz)))
# for i in range(48):
#           for j in range(1):
#                     for k in range(int(sz)):
#                               train_data[i][j][k]=xr[i][k]
test_data=np.array(test_data)
test_label=np.array([0]*14 + [1]*13 + [2]*11 + [3]*10 )
test_label=test_label.T
pred=clf.predict(test_data) 
print(accuracy_score(test_label,pred))