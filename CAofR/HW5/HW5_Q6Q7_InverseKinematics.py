#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *;
import numpy as np; 
import math;


# In[2]:


T_d=array([[0.078,-0.494,0.866,1],[0.135,-0.855,-0.500,2],[0.988,0.156,0,2],[0,0,0,1]])
Od=T_d[0:3,3]
Rx_d=T_d[0:3,0]
Ry_d=T_d[0:3,1]
Rz_d=T_d[0:3,2]


# In[3]:


def Rotation(q1,q2,q3):
    R11=cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3)
    R12=-cos(q1)*cos(q2)*sin(q3)-cos(q1)*cos(q3)*sin(q2)
    R13=sin(q1)
    R21=cos(q2)*cos(q3)*sin(q1)-sin(q1)*sin(q2)*sin(q3)
    R22=-cos(q2)*sin(q1)*sin(q3)-cos(q3)*sin(q1)*sin(q2)
    R23=-cos(q1)
    R31=cos(q2)*sin(q3)+cos(q3)*sin(q2)
    R32=cos(q2)*cos(q3)-sin(q2)*sin(q3)
    R33=0
    X=cos(q1)+cos(q1)*cos(q2)+cos(q1)*cos(q2)*cos(q3)-cos(q1)*sin(q2)*sin(q3)
    Y=sin(q1)+cos(q2)*sin(q1)-sin(q1)*sin(q2)*sin(q3)+cos(q2)*cos(q3)*sin(q1)
    Z=sin(q2)+cos(q2)*sin(q3)+cos(q3)*sin(q2)
    Rx=array([[R11],[R21],[R31]])
    Ry=array([[R12],[R22],[R32]])
    Rz=array([[R13],[R23],[R33]])
    On=array([[X],[Y],[Z]])
    return Rx,Ry,Rz,On


# In[4]:


def error(Rx,Ry,Rz,On):
    delta_O= (Od-On.ravel()).reshape(-1,1)
    delta_theta= 0.5*(np.cross(Rx.ravel(),Rx_d.ravel()) + np.cross(Ry.ravel(),Ry_d.ravel()) + np.cross(Rz.ravel(),Rz_d.ravel())).reshape(-1,1)
    error=np.vstack((delta_O,delta_theta))
    return error
    


# In[5]:


def Jacobian(q1,q2,q3, Lambda = 0.1):
  J11=sin(q1)*sin(q2)*sin(q3) - cos(q2)*sin(q1) - sin(q1) - cos(q2)*cos(q3)*sin(q1)
  J12=-cos(q1)*sin(q2)-cos(q1)*cos(q2)*sin(q3)-cos(q1)*cos(q3)*sin(q2)
  J13=-cos(q1)*cos(q2)*sin(q3)-cos(q1)*cos(q3)*sin(q2)
  J21=cos(q1) + cos(q1)*cos(q2) + cos(q1)*cos(q2)*cos(q3) - cos(q1)*sin(q2)*sin(q3)
  J22=- sin(q1)*sin(q2) - cos(q2)*sin(q1)*sin(q3) - cos(q3)*sin(q1)*sin(q2)
  J23=- cos(q2)*sin(q1)*sin(q3) - cos(q3)*sin(q1)*sin(q2)
  J31=0
  J32=cos(q2) + cos(q2)*cos(q3) - sin(q2)*sin(q3)
  J33=cos(q2)*cos(q3) - sin(q2)*sin(q3)
  J=[[J11, J12, J13],[J21, J22, J23],[J31, J32, J33],[0,0,0],[0,-1,-1],[1,0,0]]
  J = np.array(J)
 
  return J


# In[49]:


###Gradient descent
DELTA_ERROR = 1000
PREV_ERROR = 1000
q1, q2, q3 = 0,0,0
iteration = 0
while DELTA_ERROR > 0.00001:
  iteration += 1
  Rx, Ry, Rz, On = Rotation(q1, q2, q3)
  NEW_ERROR=error(Rx,Ry,Rz,On)
  qk= np.array([q1, q2, q3]) + 0.1*( (Jacobian(q1, q2, q3)).T.dot(NEW_ERROR)).ravel()
  q1=qk[0]
  q2=qk[1]
  q3=qk[2]
  DELTA_ERROR = abs(np.linalg.norm(NEW_ERROR) - PREV_ERROR)
  print(q1,q2,q3)
  PREV_ERROR = np.linalg.norm(NEW_ERROR)
  #break
  print("At the {} iteration, the error is {}".format(iteration, np.linalg.norm(NEW_ERROR)))
  print('------------------------------------')


# In[46]:


## Newton Method
Lambda= 0
DELTA_ERROR = 1000
PREV_ERROR = 1000
q1, q2, q3 = 0,0,0
iteration = 0
while DELTA_ERROR>0.0001:
  iteration += 1
  Rx, Ry, Rz, On = Rotation(q1, q2, q3)
  NEW_ERROR=error(Rx,Ry,Rz,On)
  J = Jacobian(q1, q2, q3)
  dls_pinv = np.linalg.inv(J.T @ J + Lambda**2 * np.eye(3)) @ J.T

  qk= np.array([q1, q2, q3]) +(dls_pinv.dot(NEW_ERROR)).ravel()
  # print(J_inverse(q1, q2, q3).dot(ERROR))
  # print(qk.shape)
  q1=qk[0]
  q2=qk[1]
  q3=qk[2]
  DELTA_ERROR = abs(np.linalg.norm(NEW_ERROR) - PREV_ERROR)
  print(q1,q2,q3)
  PREV_ERROR = np.linalg.norm(NEW_ERROR)
  #break
  print("At the {} iteration, the error is {}".format(iteration, np.linalg.norm(NEW_ERROR)))
  print('------------------------------------')


# In[77]:


import random
Input = []
Output = []
for _ in range(1000):
  q1 = random.uniform(-np.pi, np.pi)
  q2 = random.uniform(-np.pi, np.pi)
  q3 = random.uniform(-np.pi, np.pi)
  X=cos(q1)+cos(q1)*cos(q2)+cos(q1)*cos(q2)*cos(q3)-cos(q1)*sin(q2)*sin(q3)
  Y=sin(q1)+cos(q2)*sin(q1)-sin(q1)*sin(q2)*sin(q3)+cos(q2)*cos(q3)*sin(q1)
  Z=sin(q2)+cos(q2)*sin(q3)+cos(q3)*sin(q2)

  Output.append([q1, q2, q3])
  Input.append([X, Y, Z])

Input = np.array(Input)
Output = np.array(Output)


# In[78]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation="relu"),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3)
])



model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(Input, Output, validation_split=0.2, epochs=60, batch_size=32, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[79]:


K = 500
traj = np.zeros((K,3))
traj[:,0] = 2*np.cos(np.linspace(0,2*np.pi,num=K))
traj[:,1] = 2*np.sin(np.linspace(0,2*np.pi,num=K))
traj[:,2] = np.sin(np.linspace(0,8*np.pi,num=K))

prediction = model.predict(traj)


# In[80]:



fig = plt.figure()
ax = plt.axes(projection ='3d')
z = np.sin(np.linspace(0,8*np.pi,num=K))
x = 2*np.cos(np.linspace(0,2*np.pi,num=K))
y = 2*np.sin(np.linspace(0,2*np.pi,num=K))
 
# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot geeks for geeks')
plt.show()


# In[81]:



fig = plt.figure()
ax = plt.axes(projection ='3d')
z = prediction[:,2]
x = prediction[:,0]
y = prediction[:,1]
 
# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot geeks for geeks')
plt.show()


# In[ ]:




