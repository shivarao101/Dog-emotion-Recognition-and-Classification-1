import matplotlib.pyplot as plt
import numpy as np
#x1=np.array([66.58,64.58,68.75,72.92])
x11=np.array([693,387,434,264])
l=np.array([2,3,4,5])
#x2=np.array([70.83,66.67,66.67,77.08])
x22=np.array([28,48,19,6.46])
plt.xlabel('No of Layers-->')
plt.ylabel('Loss function-->')
plt.title('Loss function vs No of layers')
plt.plot(l,x11,'b*-',l,x22,'g*-')
plt.legend(['Without pre-processing testing loss ','With pre-processing testing loss '])