import cv2
from keras.applications import InceptionV3
import numpy as np
import argparse
from imutils import paths
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input,Dense,Flatten, AveragePooling2D,Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

from CNN import ecoCNN

from sklearn.preprocessing import LabelBinarizer
#usage
# python train_nn_scratch.py --dataset "D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT"

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",help="dataset path",required=True)
ap.add_argument("-o","--output",help="path to save the output model",default="nn_ecographic_prediction.h5")

args=vars(ap.parse_args())

EPOCHS=40
BS=6
INIT_LR=5e-4
undersample=False


def load_img(path,target_dim):
    """Load, grayscale and resize the img"""
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.resize(img,target_dim)/255.


#0. load data


print(f"[INFO] Loading dataset..")

classes=os.listdir(args["dataset"])

X=[]
y=[]
for c in classes:
    im_list=paths.list_images(os.path.join(args["dataset"],c))
    for i in im_list:
        if "mask" not in i:
            X.append(load_img(i,(250,250)))
            y.append(c)


print(f"[INFO] loading completed len X: {len(X)}\t len y: {len(y)}")


print(f"[INFO] Starting undersampling procedure..")
unique, counts = np.unique(y, return_counts=True)
print(unique,counts)
min=min(counts)



if undersample:
    u_X=X[:min]+X[counts[0]:counts[0]+min]+X[counts[0]+counts[1]:counts[0]+counts[1]+min]
    u_y=y[:min]+y[counts[0]:counts[0]+min]+y[counts[0]+counts[1]:counts[0]+counts[1]+min]

    u_unique, u_counts = np.unique(u_y, return_counts=True)
    print("[INFO] After undersampling procedure, balancing classes..")
    print(u_unique,u_counts)

    X=u_X
    y=u_y
#1 build the model
X=np.expand_dims(X,-1)


model=ecoCNN.build_model(len(classes),X[0].shape)

model.summary()

print(f"[INFO] compiling model..")
opt=Adam(learning_rate=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])


#prepare the data


X,y=shuffle(X,y)
lb=LabelBinarizer()

y=np.array(lb.fit_transform(y))
y=to_categorical(y)
X=np.array(X)
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20,stratify=y)


print(f"[INFO] Check dimension pre-training:\n x_train : {x_train.shape}\t x_test : {x_test.shape}\n y_train : {y_train.shape}\t y_test : {y_test.shape}")

#x_train=np.reshape(x_train,(len(x_train),224,224,3))
#x_test=np.reshape(x_test,(len(x_test),224,224,3))

#2. prepare the model, set trainable layers
trainAug=ImageDataGenerator(rotation_range=8,fill_mode="nearest",brightness_range=[0,1.],zoom_range=0.2,horizontal_flip=True,shear_range=0.3)


print(f"[INFO] training the model..")

#LAVORARE QUA AGGIUNGERE CHECKPOINT -> SALVARE LABEL

H = model.fit_generator(
	trainAug.flow(x_train, y_train, batch_size=BS),
	steps_per_epoch=len(x_train) // BS,
	validation_data=(x_test, y_test),
	validation_steps=len(x_test) // BS,
	epochs=EPOCHS)
print(f"[INFO] evaluating the network..")
predIdx=model.predict(x_test,batch_size=BS)

predIdx=np.argmax(predIdx,axis=1) #get the max

print(classification_report(y_test.argmax(axis=1),predIdx,target_names=lb.classes_))

cm=confusion_matrix(y_test.argmax(axis=1),predIdx)
total=sum(sum(cm))
acc=(cm[0,0]+cm[1,1])/total

sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
specifity=cm[1,1]/(cm[1,0]+cm[1,1])

print(cm)
print(f"acc: {round(acc,5)}")
print(f"sensitivity: {round(sensitivity,5)}")
print(f"specifity: {round(specifity,5)}")
