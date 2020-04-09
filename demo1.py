from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
import cv2
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt     
from keras.preprocessing import image                  
#from tqdm import tqdm
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import log_loss
import math
import pickle
import os
from keras.optimizers import Adam

def load_train (path):
    data=load_files(path)
    driver_file=np.array(data['filenames'])
    driver_target=np_utils.to_categorical(np.array(data['target']),10)
    
    return driver_file,driver_target

def load_test(path):
    data=glob(path)
    driver_file=np.array(glob(path))
    driver_id=[]
    for p in data:
        driver_id.append(os.path.basename(p))
   # driver_id=os.path.basename(p)
    #test_id=np.array(driver_file['target'])
    return driver_file,driver_id

#training set - separated by classes
train_files,train_targets = load_train('imgs/train')

#test set - has no labels
test_files,test_id = load_test('imgs/test/*')


#split the training set 
X_train, X_test, y_train, y_test = train_test_split(train_files,train_targets ,test_size=0.3,random_state=42)

#Pre-process data
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)                

# pre-process the data for Keras
train_targets=y_train
test_targets=y_test
train_tensors = paths_to_tensor(X_train).astype('float32')/255
test_tensors = paths_to_tensor(X_test).astype('float32')/255

#Training model 
model = Sequential()
print('x_train shape:', train_tensors.shape)
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model callbacks 
checkpointer = ModelCheckpoint(filepath='saved_models/weightsfromscratch.hdf5', 
                               verbose=1, save_best_only=True)

history=model.fit(train_tensors, train_targets, validation_split=0.2,epochs=5, batch_size=20, callbacks=[checkpointer], verbose=1)

#model accuracy 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






