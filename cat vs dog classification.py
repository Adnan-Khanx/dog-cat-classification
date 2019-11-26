#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt


# In[2]:


classifier=Sequential()


# In[3]:


classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))


# In[4]:


classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[5]:


classifier.add(Flatten())


# In[6]:


classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))


# In[7]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[8]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(
    'D:/sample Projects/dog and cat/Convolutional_Neural_Networks/dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

test_set=test_datagen.flow_from_directory(
    'D:/sample Projects/dog and cat/Convolutional_Neural_Networks/dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')


# In[9]:


from IPython.display import display
from PIL import Image
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=3,
    validation_data=test_set,
    validation_steps=800)


# In[10]:


import numpy as np
from keras.preprocessing import image
test_image=image.load_img('D:/sample Projects/dog and cat/Convolutional_Neural_Networks/dataset/test_set/cats/cat.4006.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
    prediction='dog'
else:
    prediction='cat'
  

print(prediction)

