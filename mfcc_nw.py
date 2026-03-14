import numpy as np
import scipy
def dct(x):
    N=len(x)
    X=np.zeros(N)
    for k in range(N):
        for n in range(N):
            X[k]=X[k]+ (np.cos(((np.pi*(2*n+1)*k)/(2*N)))*x[n])
        if k==0:
            X[k]=np.sqrt(1/N)*X[k]
        else:
            X[k]=np.sqrt(2/N)*X[k]
    return X
def triang(N):
    w=[0]*N
    for n in range(N):
        w[n]=1- 2*(np.abs((n-((N-1)/2))/(N-1)))
    return w
def melcepstrum(a,fs,frame_length,frame_step,filter_bank_size):
    l=0
    x=[]
    for i in range(0,len(a)-frame_length,frame_step):
        for k in range(frame_length):
            x.append(a[i+k])
            l+=1
    if len(x)%2==0:
        cols=len(x)/frame_length
    else:
        x=x+[0]
        cols=len(x)/frame_length
    cols=int(cols)
    x1=np.reshape(x,(frame_length,cols),order='F')
    x2=np.zeros((frame_length,cols))
    for i in range(cols):
        if(list(x1[:,i])==[0]*len(x1[:,i])):
            for j in range(1):
                x1[j][i]=0.0000000001
    c=np.hamming(frame_length)
    for i in range(cols):
        x2[:,i]=x1[:,i]*c
    fx=np.zeros((frame_length,cols))
    for i in range(cols):
        fx[:,i]=(1/frame_length)*(np.abs(np.fft.fft(x2[:,i]))**2)
    fx=fx[0:int(frame_length/2),0:cols]
    fl=300
    fu=fs/2
    f=[i for i in np.arange(fl,fu)]
    f=np.array(f)
    mel=1125*(np.log(1+f/700))
    step=(mel[len(mel)-1]-mel[0])/(filter_bank_size+2)
    fmel=[i for i in np.arange(mel[0],mel[len(mel)-1]+1,step)]
    fmel=np.array(fmel)
    fin=700*(np.exp((fmel)/(1125))-1)
    rn=np.floor(((frame_length+1)*fin)/(fs))
    rn=list(rn)
    mel_win=list()
    for i in range(len(rn)-2):
        mel_win.append([0]*int(rn[i]-1)+triang(int(rn[i+2])-int(rn[i]))+[0]*(frame_step-int(rn[i+2])+1))
    mel_win=np.array(mel_win)
    mel_win=np.transpose(mel_win)
    fx1=np.zeros((filter_bank_size,cols))
    for i in range(cols):
        for k in range(filter_bank_size):
            fx1[k,i]=sum(fx[:,i]*mel_win[:,k])
    fx4=np.zeros((filter_bank_size,cols))
    fxm=np.where(fx1>0.000000001,fx1,-0.01)
    fxm=np.where(np.isnan(fxm),0,fxm)
    fx3=np.log10(fxm,out=fxm,where=fxm>0)
    #fx3=np.log(fx1)
    for i in range(cols):
        fx4[:,i]=scipy.fft.dct(fx3[:,i])
    return fx4