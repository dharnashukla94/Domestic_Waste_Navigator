
# coding: utf-8

# In[1]:


import pickle


# In[2]:


import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import pandas as pd
import zipfile as zf
import shutil
import re
import seaborn as sns
from random import shuffle
# Import Image manipulation
import PIL.Image


# Import PyTorch
#import torch
#from torch import nn
#import torch.nn.functional as F
#import torchvision.transforms.functional as TF
#from torch.utils.data import Dataset, DataLoader
#import albumentations as A
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import multilabel_confusion_matrix
# Keras Imports

import keras
import tensorflow 
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation , Dropout
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam , RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121


# In[4]:


#from zipfile import ZipFile
#with ZipFile('data.zip', 'r') as zipObj:
#    zipObj.extractall()


# In[5]:


# show data 

w=20
h=20
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5

for i in range(20):
    img =PIL.Image.open("final_dataset/Blue_bin/Blue_bin{}.png".format(i+1))
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img)  
    
plt.show()
#plt.title('Contains of Blue Bin ')


# In[9]:


# Rearranging data in different sets

# Random shuffling of images
dirs = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin']
subtype = ['train','test','valid']
folder = "final_dataset/"
for names in dirs:
    source_folder = os.path.join(folder,names)
    files = os.listdir(source_folder)
    shuffle(files)
    num = len(files)
    train_len = int(num*0.5)
    valid_len = int(num*0.25)
    test_len = num - train_len - valid_len
    print("dist for {} is train: {} test: {} valid: {} and total images: {} ".format(names, train_len, test_len, valid_len, train_len+test_len+valid_len))
    path = "data"
    if not os.path.exists(path):
        os.makedirs(path)
    path1 = "data/train"
    if not os.path.exists(path1):
        os.makedirs(path1)
    
    path2 = "data/test"
    if not os.path.exists(path2):
        os.makedirs(path2) 
    path3 = "data/valid"
    if not os.path.exists(path3):
        os.makedirs(path3)  
        
    if not os.path.exists("data/train/"+ names):    
        os.makedirs("data/train/"+ names)
        
    if not os.path.exists("data/valid/"+ names):    
        os.makedirs("data/valid/"+ names)    
        
    if not os.path.exists("data/test/"+ names):    
        os.makedirs("data/test/"+ names)    
        
    for i in range(train_len):
        src = source_folder+'/'+files[i]
        dst = "data/train"+'/'+ names+'/'+files[i]
        shutil.copy(src, dst)
        
    for i in range(train_len, train_len+valid_len):
        src = source_folder+'/'+files[i]
        dst = "data/valid"+'/'+ names+'/'+files[i]
        shutil.copy(src, dst) 
        
    for i in range(train_len+valid_len,num):
        src = source_folder+'/'+files[i]
        dst = "data/test"+'/'+ names+'/'+files[i]
        shutil.copy(src, dst)    


# In[53]:


def plot_accuracy_graph(model):
    acc =model.history['acc']
    val_acc = model.history['val_acc']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy' , color = 'r')
    plt.plot(val_acc, label='Validation Accuracy' , color = 'g')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')


# In[52]:


def plot_loss(model):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    loss =model.history['loss']
    val_loss = model.history['val_loss']
    plt.plot(loss, label='Training Loss' , color = 'r')
    plt.plot(val_loss, label='Validation Loss' ,color = 'g')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    #plt.ylim([0,12.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# # Non Normalized Dataset

# In[6]:


train_datagen_ = ImageDataGenerator(rescale = 1./1.)

test_datagen_ = ImageDataGenerator()


# In[7]:


train_set_ = train_datagen_.flow_from_directory('data/train', 
                                                     target_size = (150,150) , 
                                                     classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] ,
                                                     class_mode="binary",
                                                     batch_size=32
                                                                  )
valid_set_ = test_datagen_.flow_from_directory('data/valid', target_size = (150,150) , classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] , class_mode="binary" ,batch_size=32 , shuffle = False)
test_set_ = test_datagen_.flow_from_directory('data/test', target_size = (150,150) , classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] ,class_mode="binary" ,batch_size=32 ,shuffle = False )


# In[9]:


train_set_.classes


# # VGG16 Model

# In[23]:


# Training VGG16 for the dataset

IMG_HEIGHT = 150
IMG_WIDTH = 150

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = vgg.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
vgg = Model(vgg.input, output)
for layer in vgg.layers:
    layer.trainable = False
vgg.summary()


# In[24]:


input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model = Sequential()
model.add(vgg)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()


# In[25]:


history = model.fit_generator(train_set_, 
                              steps_per_epoch=30, 
                              epochs=100,
                              validation_data=valid_set_, 
                              validation_steps=50, 
                              verbose=1)


# In[26]:


# saving VGG16 model

model.save('VGG16.h5')


# In[5]:


# loading model

#model = tensorflow.keras.models.load_model('VGG16.h5')

# Show the model architecture
#model.summary()


# In[8]:



score_train = model.evaluate_generator(train_set_,verbose=0)
score_test = model.evaluate_generator(test_set_, verbose=0)
score_valid = model.evaluate_generator(valid_set_, verbose=0)

print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1]*100)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1]*100)
print('validation loss:', score_valid[0])
print('validation accuracy:', score_valid[1]*100)


# In[32]:


plot_accuracy_graph(history)# VGG16


# In[33]:


plot_loss(history) # VGG16


# In[34]:


y_true = test_set_.classes ## y_true

Y_pred = model.predict_generator(generator=test_set_)
y_pred = np.argmax(Y_pred, axis=1)


# In[35]:


cnf = multilabel_confusion_matrix(y_true,y_pred)
print('Classification Report')
print(classification_report(y_true, y_pred, labels=[0,1,2,3]))


# In[113]:


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=['Black bin','Blue bin','Green bin','Regular garbage'], index = ['Black bin','Blue bin','Green bin','Regular garbage'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15} , fmt='2g')


# In[112]:


x,y = test_set_.next()
w=20
h=20
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5

new_x = x[0:20]
new_y = y[0:20]

img_pred= model.predict_classes(new_x)
names = ['Black bin','Blue bin','Green bin','Regular garbage']
for i in range(0,20):
    img =new_x[i]*255
    img_class = img_pred[i]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(names[img_class] ,fontsize=16)
    plt.imshow((img).astype(np.uint8) ,interpolation=None)  


# In[144]:


count = 0
for i in range(0,20):
    if new_y[i] != img_pred[i]: 
        count=count+1
        
print("Misclassified {} images out of {} images".format(count, len(new_y)))


# # Resnet 50 Model

# In[123]:


# Training Resnet50 for the dataset

IMG_HEIGHT = 150
IMG_WIDTH = 150

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = resnet.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)
for layer in resnet.layers:
    layer.trainable = False
resnet.summary()


# In[124]:


input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model1 = Sequential()
model1.add(resnet)
model1.add(Dense(512, activation='relu', input_dim=input_shape))
model1.add(Dropout(0.3))
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(4, activation='softmax'))

model1.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model1.summary()


# In[125]:


history1 = model1.fit_generator(train_set_, 
                              steps_per_epoch=30, 
                              epochs=100,
                              validation_data=valid_set_, 
                              validation_steps=50, 
                              verbose=1)


# In[15]:


score_train = model1.evaluate_generator(train_set_,verbose=0)
score_test = model1.evaluate_generator(test_set_, verbose=0)
score_valid = model1.evaluate_generator(valid_set_, verbose=0)

print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1]*100)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1]*100)
print('validation loss:', score_valid[0])
print('validation accuracy:', score_valid[1]*100)


# In[127]:


plot_accuracy_graph(history1)# Resnet50


# In[128]:


plot_loss(history1) # Resnet50


# In[129]:


y_true = test_set.classes ## y_true

Y_pred = model1.predict_generator(generator=test_set_)
y_pred = np.argmax(Y_pred, axis=1)

cnf1 = multilabel_confusion_matrix(y_true,y_pred)
print('Classification Report')
print(classification_report(y_true, y_pred, labels=[0,1,2,3]))


# In[131]:


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=['Black bin','Blue bin','Green bin','Regular garbage'], index = ['Black bin','Blue bin','Green bin','Regular garbage'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16} , fmt='2g')# font size


# In[132]:


# saving ResNet50 model

model1.save('Resnet50.h5')


# In[6]:


# # loading model

#model1 = tensorflow.keras.models.load_model('/Users/dharnashukla/Desktop/Final_Project/Resnet50.h5')

# # Show the model architecture
# model1.summary()


# In[185]:



x,y = test_set_.next()
w=20
h=20
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5

new_x = x[0:20]
new_y = y[0:20]

img_pred= model1.predict_classes(new_x)
names = ['Black bin','Blue bin','Green bin','Regular garbage']
for i in range(0,20):
    img =new_x[i]*255
    img_class = img_pred[i]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(names[img_class] ,fontsize=16)
    plt.imshow((img).astype(np.uint8) ,interpolation=None)  


# In[186]:


count = 0
for i in range(0,20):
    if new_y[i] != img_pred[i]: 
        count=count+1
        
print("Misclassified {} images out of {} images".format(count, len(new_y)))


#  # Normalized Dataset

# In[7]:


train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[8]:


train_set = train_datagen.flow_from_directory('data/train', 
                                                     target_size = (150,150) , 
                                                     classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] ,
                                                     class_mode="binary",
                                                     batch_size=32
                                                                  )
valid_set = test_datagen.flow_from_directory('data/valid', target_size = (150,150) , classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] , class_mode="binary" ,batch_size=32 , shuffle = False)
test_set = test_datagen.flow_from_directory('data/test', target_size = (150,150) , classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] ,class_mode="binary" ,batch_size=32 ,shuffle = False )


# # Mobile net Model

# In[8]:


# Traning mobile net for the Dataset 

IMG_HEIGHT = 150
IMG_WIDTH = 150

mobilenet = MobileNet(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = mobilenet.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
mobilenet = Model(mobilenet.input, output)
for layer in mobilenet.layers:
    layer.trainable = False
mobilenet.summary()


# In[9]:


input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model2 = Sequential()
model2.add(mobilenet)
model2.add(Dense(512, activation='relu', input_dim=input_shape))
model2.add(Dropout(0.3))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(4, activation='softmax'))

model2.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model2.summary()


# In[10]:


history2 = model2.fit_generator(train_set, 
                              steps_per_epoch=30, 
                              epochs=100,
                              validation_data=valid_set, 
                              validation_steps=50, 
                              verbose=1)


# In[11]:


score_train = model2.evaluate_generator(train_set,verbose=0)
score_test = model2.evaluate_generator(test_set, verbose=0)
score_valid = model2.evaluate_generator(valid_set, verbose=0)

print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1]*100)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1]*100)
print('validation loss:', score_valid[0])
print('validation accuracy:', score_valid[1]*100)


# In[19]:


plot_accuracy_graph(history2)


# In[20]:


plot_loss(history2)


# In[21]:


y_true = test_set.classes ## y_true

Y_pred = model2.predict_generator(generator=test_set)
y_pred = np.argmax(Y_pred, axis=1)

cnf1 = multilabel_confusion_matrix(y_true,y_pred)
print('Classification Report')
print(classification_report(y_true, y_pred, labels=[0,1,2,3]))


# In[22]:


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=['Black bin','Blue bin','Green bin','Regular garbage'], index = ['Black bin','Blue bin','Green bin','Regular garbage'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16} , fmt='2g')# font size


# In[23]:


# saving mobilenet model

model2.save('MobileNet.h5')


# In[25]:


with open('History_MobileNet.pkl', 'wb') as file_i:
        pickle.dump(history2.history, file_i)


# In[16]:


# loading model

#model2 = tensorflow.keras.models.load_model('MobilenetV2.h5')

# Show the model architecture
#model2.summary()


# In[26]:


x,y = test_set.next()
w=20
h=20
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5

new_x = x[0:20]
new_y = y[0:20]

img_pred= model2.predict_classes(new_x)
names = ['Black bin','Blue bin','Green bin','Regular garbage']
for i in range(0,20):
    img =new_x[i]*255
    img_class = img_pred[i]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(names[img_class] ,fontsize=16)
    plt.imshow((img).astype(np.uint8) ,interpolation=None)  


# In[27]:


count = 0
for i in range(0,20):
    if new_y[i] != img_pred[i]: 
        count=count+1
        
print("Misclassified {} images out of {} images".format(count, len(new_y)))


# # VGG19 Model

# In[30]:


IMG_HEIGHT = 150
IMG_WIDTH = 150

net = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = net.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
net = Model(net.input, output)
for layer in net.layers:
    layer.trainable = False
net.summary()


# In[31]:


input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model3 = Sequential()
model3.add(net)
#model.Flatten()
model3.add(Dense(512, activation='relu', input_dim=input_shape))
model3.add(Dropout(0.3))
model3.add(BatchNormalization())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(4, activation='softmax'))

model3.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model3.summary()


# In[32]:


history3 = model3.fit_generator(train_set, 
                              steps_per_epoch=32, 
                              epochs=100,
                              validation_data=valid_set, 
                              validation_steps=50, 
                              verbose=1)


# In[33]:


with open('History_VGG19.pkl', 'wb') as file_i:
        pickle.dump(history3.history, file_i)


# In[34]:


score_train = model3.evaluate_generator(train_set,verbose=0)
score_test = model3.evaluate_generator(test_set, verbose=0)
score_valid = model3.evaluate_generator(valid_set, verbose=0)

print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1]*100)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1]*100)
print('validation loss:', score_valid[0])
print('validation accuracy:', score_valid[1]*100)


# In[54]:


plot_accuracy_graph(history3)


# In[55]:


plot_loss(history3)


# In[37]:


y_true = test_set.classes ## y_true

Y_pred = model3.predict_generator(generator=test_set)
y_pred = np.argmax(Y_pred, axis=1)

cnf1 = multilabel_confusion_matrix(y_true,y_pred)
print('Classification Report')
print(classification_report(y_true, y_pred, labels=[0,1,2,3]))


# In[38]:


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=['Black bin','Blue bin','Green bin','Regular garbage'], index = ['Black bin','Blue bin','Green bin','Regular garbage'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16} , fmt='2g')# font size


# In[39]:


# saving mobilenet model

model3.save('v6619.h5')


# In[42]:


# loading model

#model3 = tensorflow.keras.models.load_model('v6619.h5')

# # Show the model architecture
# model3.summary()


# In[40]:


x,y = test_set.next()
w=20
h=20
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5

new_x = x[0:20]
new_y = y[0:20]

img_pred= model3.predict_classes(new_x)
names = ['Black bin','Blue bin','Green bin','Regular garbage']
for i in range(0,20):
    img =new_x[i]*255
    img_class = img_pred[i]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(names[img_class] ,fontsize=16)
    plt.imshow((img).astype(np.uint8) ,interpolation=None)  


# In[41]:


print("Misclassified {} images out of {} images".format(len(set(new_y).intersection(img_pred)), len(new_y)))


# # Normalized Dataset With Differebnt Image_Size

# In[59]:


train_datagenD = ImageDataGenerator(rescale = 1./255)

test_datagenD = ImageDataGenerator(rescale = 1./255)


# In[68]:


train_set_D = train_datagenD.flow_from_directory('data/train', 
                                                     target_size = (224,224) , 
                                                     classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] ,
                                                     class_mode="binary",
                                                     batch_size=32
                                                                  )
valid_set_D = test_datagenD.flow_from_directory('data/valid', target_size = (224,224) , classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] , class_mode="binary" ,batch_size=32 , shuffle = False)
test_set_D = test_datagenD.flow_from_directory('data/test', target_size = (224,224) , classes = ['Black_bin' , 'Blue_bin' ,'Green_bin' , 'Regular_bin'] ,class_mode="binary" ,batch_size=32 ,shuffle = False )


# # DenseNet 121

# In[69]:


IMG_HEIGHT = 224
IMG_WIDTH = 224

net1 = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = net1.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
net1 = Model(net1.input, output)
for layer in net1.layers:
    layer.trainable = False
net1.summary()


# In[70]:


input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model4 = Sequential()
model4.add(net1)
#model.Flatten()
model4.add(Dense(512, activation='relu', input_dim=input_shape))
model4.add(Dropout(0.5))
model4.add(BatchNormalization())
model4.add(Dense(512, activation='relu'))
model4.add(Dropout(0.5))
model4.add(BatchNormalization())
#model4.add(Dense(128, activation='relu'))
#model4.add(Dropout(0.5))
model4.add(Dense(4, activation='softmax'))

model4.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model4.summary()


# In[71]:


history4 = model4.fit_generator(train_set_D, 
                              steps_per_epoch=32, 
                              epochs=50,
                              validation_data=valid_set_D, 
                              validation_steps=50, 
                              verbose=1)


# In[72]:


score_train = model4.evaluate_generator(train_set_D,verbose=0)
score_test = model4.evaluate_generator(test_set_D, verbose=0)
score_valid = model4.evaluate_generator(valid_set_D, verbose=0)

print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1]*100)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1]*100)
print('validation loss:', score_valid[0])
print('validation accuracy:', score_valid[1]*100)


# In[73]:


plot_accuracy_graph(history4)


# In[74]:


plot_loss(history4)


# In[75]:


y_true = test_set.classes ## y_true

Y_pred = model4.predict_generator(generator=test_set_D)
y_pred = np.argmax(Y_pred, axis=1)

cnf2 = multilabel_confusion_matrix(y_true,y_pred)
print('Classification Report')
print(classification_report(y_true, y_pred, labels=[0,1,2,3]))


# In[76]:


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=['Black bin','Blue bin','Green bin','Regular garbage'], index = ['Black bin','Blue bin','Green bin','Regular garbage'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16} , fmt='2g')# font size


# In[77]:


# saving mobilenet model

model4.save('DenseNet121.h5')


# In[39]:


# loading model

#model5 = tensorflow.keras.models.load_model('InceptionV3.h5')

# # Show the model architecture
# model5.summary()


# In[81]:


x,y = test_set_D.next()
w=20
h=20
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5

new_x = x[0:20]
new_y = y[0:20]

img_pred= model4.predict_classes(new_x)
names = ['Black bin','Blue bin','Green bin','Regular garbage']
for i in range(0,20):
    img =new_x[i]*255
    img_class = img_pred[i]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(names[img_class] ,fontsize=16)
    plt.imshow((img).astype(np.uint8) ,interpolation=None) 


# In[79]:


print("Misclassified {} images out of {} images".format(len(set(new_y).intersection(img_pred)), len(new_y)))


# In[82]:


with open('History_DenseNet.pkl', 'wb') as file_D:
        pickle.dump(history4.history, file_D)


# # InceptionV3 Net 

# In[85]:


IMG_HEIGHT = 224
IMG_WIDTH = 224

net2 = InceptionV3(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = net2.layers[-1].output
output = tensorflow.keras.layers.Flatten()(output)
net2 = Model(net2.input, output)
for layer in net2.layers:
    layer.trainable = False
net2.summary()


# In[88]:


input_shape=(IMG_HEIGHT,IMG_WIDTH,3)

model5 = Sequential()
model5.add(net2)
#model.Flatten()
model5.add(Dense(512, activation='relu', input_dim=input_shape))
model5.add(Dropout(0.5))
model5.add(Dense(512, activation='relu'))
model5.add(Dropout(0.5))
model5.add(Dense(4, activation='softmax'))

model5.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model5.summary()


# In[89]:


history5 = model5.fit_generator(train_set_D, 
                              steps_per_epoch=30, 
                              epochs=100,
                              validation_data=valid_set_D, 
                              validation_steps=50, 
                              verbose=1)


# In[90]:


score_train = model5.evaluate_generator(train_set_D,verbose=0)
score_test = model5.evaluate_generator(test_set_D, verbose=0)
score_valid = model5.evaluate_generator(valid_set_D, verbose=0)

print('Train loss:', score_train[0])
print('Train accuracy:', score_train[1]*100)
print('Test loss:', score_test[0])
print('Test accuracy:', score_test[1]*100)
print('validation loss:', score_valid[0])
print('validation accuracy:', score_valid[1]*100)


# In[91]:


plot_accuracy_graph(history5)


# In[92]:


plot_loss(history5)


# In[93]:


y_true = test_set.classes ## y_true

Y_pred = model5.predict_generator(generator=test_set_D)
y_pred = np.argmax(Y_pred, axis=1)

cnf2 = multilabel_confusion_matrix(y_true,y_pred)
print('Classification Report')
print(classification_report(y_true, y_pred, labels=[0,1,2,3]))


# In[94]:


data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=['Black bin','Blue bin','Green bin','Regular garbage'], index = ['Black bin','Blue bin','Green bin','Regular garbage'])
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16} , fmt='2g')# font size


# In[95]:


# saving mobilenet model

model5.save('InceptionV3.h5')


# In[96]:


# loading model

#model5 = tensorflow.keras.models.load_model('InceptionV3.h5')

# # Show the model architecture
# model5.summary()


# In[97]:


x,y = test_set_D.next()
w=20
h=20
fig=plt.figure(figsize=(12, 12))
columns = 4
rows = 5

new_x = x[0:20]
new_y = y[0:20]

img_pred= model5.predict_classes(new_x)
names = ['Black bin','Blue bin','Green bin','Regular garbage']
for i in range(0,20):
    img =new_x[i]*255
    img_class = img_pred[i]
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.title(names[img_class] ,fontsize=16)
    plt.imshow((img).astype(np.uint8) ,interpolation=None) 


# In[98]:


print("Misclassified {} images out of {} images".format(len(set(new_y).intersection(img_pred)), len(new_y)))


# In[99]:


with open('History_Inception.pkl', 'wb') as file_i:
        pickle.dump(history5.history, file_i)

