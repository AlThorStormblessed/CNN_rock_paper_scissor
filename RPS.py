import collections.abc

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from PIL import Image
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import glob
import random
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.utils import to_categorical

print("Imports done")

Images = []

rocks = []
for file in tqdm(glob.glob("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Rock_paper_scissors/image_data/rock/*jpg")):
    rocks_ = []
    imd = cv2.imread(file)
    imd = cv2.cvtColor(imd, cv2.COLOR_BGR2RGB)
    imd = cv2.resize(imd, (224, 224))
    imd = imd/255
    rocks_.append(imd)
    rocks_.append(0)
    rocks.append(rocks_)

Images.extend(rocks)

paper = []
for file in tqdm(glob.glob("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Rock_paper_scissors/image_data/paper/*jpg")):
    paper_ = []
    imd = cv2.imread(file)
    imd = cv2.cvtColor(imd, cv2.COLOR_BGR2RGB)
    imd = cv2.resize(imd, (224, 224))
    imd = imd/255
    paper_.append(imd)
    paper_.append(1)
    paper.append(paper_)

Images.extend(paper)

scissors = []
for file in tqdm(glob.glob("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Rock_paper_scissors/image_data/scissors/*jpg")):
    scissors_ = []
    imd = cv2.imread(file)
    imd = cv2.cvtColor(imd, cv2.COLOR_BGR2RGB)
    imd = cv2.resize(imd, (224, 224))
    imd = imd/255
    scissors_.append(imd)
    scissors_.append(2)
    scissors.append(scissors_)

Images.extend(scissors)

none = []
for file in tqdm(glob.glob("C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Rock_paper_scissors/image_data/none/*jpg")):
    none_ = []
    imd = cv2.imread(file)
    imd = cv2.cvtColor(imd, cv2.COLOR_BGR2RGB)
    imd = cv2.resize(imd, (224, 224))
    imd = imd/255
    none_.append(imd)
    none_.append(3)
    none.append(none_)

Images.extend(none)

X, y = zip(*Images)
y = to_categorical(y, dtype ="uint8")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

def get_model():
    
    model=Sequential()
    
    base_model = MobileNet(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(224, 224, 3),
        classes=4,
        pooling='avg',
        include_top=False
    )
    #freeezing the weights of the final layer 
    for layer in base_model.layers:
        layer.trainable=False
        
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax')) #final op layer
    
    return model

model = get_model()
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary)

model.fit(x=np.array(X_train),y=np.array(y_train), batch_size=32, 
          validation_data=(np.array(X_test),np.array(y_test)),
          epochs=20)
model.save("saved_model/model_t")