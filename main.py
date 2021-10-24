import os
import numpy as np
from tensorflow.keras.layers import Dense,Flatten,Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
path='C:\\Users\\Admin\\Desktop\\pythonProject\\Flower\\flowers'
#Get data
label=[]
pixel=[]

for category in os.listdir(path):
    for flower in os.listdir(os.path.join(path,category)):
        label.append(category.split('\\')[-1])
        image=cv2.imread(os.path.join(os.path.join(path,category,flower)))
        pixel.append(cv2.resize(image,dsize=(128,128)))

label=np.array(label).reshape(-1,1)

label=LabelEncoder().fit_transform(label)
pixel=np.array(pixel)

#seperate data into train and test

X_train, X_test,y_train,y_test=train_test_split(pixel,label,train_size=0.33,shuffle=True,random_state=42)
y_train=to_categorical(y_train,num_classes=5)
y_test=to_categorical(y_test,num_classes=5)


#create model

model=VGG16(weights='imagenet',include_top=False)
for layer in model.layers:
    layer.trainable=False

input=Input(shape=(128,128,3),name='image_input')
output_vgg16=model(input)

x = Flatten(name='flatten')(output_vgg16)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(5, activation='softmax', name='predictions')(x)

my_model = Model(inputs=input, outputs=x)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

file="weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

my_model.fit(X_train, y_train, batch_size=32,
                               epochs=50,
                               validation_data=(X_test,y_test),
                               callbacks=callbacks_list)

my_model.save("vggmodel.h5")

