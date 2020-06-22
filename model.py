import numpy as np
import os
import sklearn
import matplotlib.image as mpimg
import tensorflow as tf
import pandas as pd
import keras
import cv2
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D

d=pd.read_csv('/opt/carnd_p3/data/driving_log.csv')
img_path='/opt/carnd_p3/data/'

#Function to augument and generate additional images
def aug_data(row):
    
    generated_data = {}#Dictionary to hold all the images and corrosponding angles
   
    center = mpimg.imread(img_path+row['center'].strip())
    center_angle = float(row['steering'])
    generated_data['center'] = (center, center_angle)
    
    left_image = mpimg.imread(img_path+row['left'].strip())
    left_angle =min(center_angle + .2,1)
    generated_data['left'] = (left_image, left_angle)
    
    right_image = mpimg.imread(img_path+row['right'].strip())
    right_angle = max(center_angle - .2,-1)
    generated_data['right'] = (right_image, right_angle)
    
    if center_angle!=0:#Flipping of images is required only when the there is steering , this is for adjusting the angles by 0.2 on either side to avoid vehicle from going off track
        flip = cv2.flip(center,1)
        flipped_angle = center_angle * -1
        generated_data['flipped'] = (flip, flipped_angle)
    
    
    return generated_data # Center,Left,Right and Flipped images are returned
    
#X->Images
#y->Steering angles
X=[]
y=[]
for indx,row in d.iterrows():#iterrating through each row in the driving log
    generated_data = aug_data(row)# generating additional data by flipping images and adjusting angles
    for image, angle in generated_data.values():# looping through all the generated images and storing images and angles in arrays X and y
        X.append(image)
        y.append(angle)

#Making arrays numpy to use in the model
X= np.array(X)
y= np.array(y)

#CNN built using NVIDIA's model as baseline
#Modifications include :-
#Usage of ELU instead of RELU activation as it can produce negative outputs
#Cropping of image to isolate only the road portion
#Dropout before flattening to avoid overfitting
model = keras.models.Sequential([
    keras.layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)),#Normalized Input Layer
    keras.layers.Cropping2D(cropping=((70,25),(0,0))),
    keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation="elu"),#Convolutional Layers
    keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation="elu"),
    keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation="elu"),
    keras.layers.Conv2D(64, (3, 3), activation="elu"),
    keras.layers.Conv2D(64, (3, 3), activation="elu"),
    keras.layers.Dropout(0.5),#To avoid overfitting model to training set data
    keras.layers.Flatten(),
    keras.layers.Dense(1164),#Fully Connected Layers
    keras.layers.Dense(100),python
    keras.layers.Dense(50),
    keras.layers.Dense(10),
    keras.layers.Dense(1)#Output Layer
])

model.compile(loss='mse',
              optimizer=Adam(lr=1e-3),
              metrics=['accuracy'])

model.fit(X, y, epochs=5, validation_split=0.3)

model.summary()

model.save('model.h5')
