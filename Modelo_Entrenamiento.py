"""
Medina Cadena Brandy Berlin
 Red neuronal convoluvional
14/Mayo/2023
"""

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

train = keras.utils.image_dataset_from_directory('Expressions/train/',
                                                  labels="inferred",
                                                  label_mode="categorical",
                                                  color_mode="rgb",
                                                  batch_size=32,
                                                  image_size=(50,50))

test = keras.utils.image_dataset_from_directory('Expressions/test/',
                                                labels="inferred",
                                                label_mode="categorical",
                                                color_mode="rgb",
                                                batch_size=32,
                                                image_size=(50,50))


model = Sequential()

model.add(Conv2D(250,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=(50,50,3)))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(125,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(65,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(250,activation='relu'))
model.add(Dense(125,activation='relu'))
model.add(Dense(65,activation='relu'))

model.add(Dense(8,activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.summary()

model = keras.models.load_model('CNN_Expressions_3.h5')
hist = model.fit(train,batch_size=150,epochs=5,verbose=1,validation_data=test)


model.save('CNN_Expressions_4.h5')
#model = keras.models.load_model('CNN_Expressions.h5')
