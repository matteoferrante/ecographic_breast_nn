""" Rete InceptionV3 per feature extraction + classificatori -> viene bene"""
import pickle

import cv2
from tensorflow.keras.applications import InceptionV3
import numpy as np
import argparse
from imutils import paths
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input,Dense,Flatten, AveragePooling2D,Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.utils import shuffle

from sklearn import svm
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#usage
# python inception_rf.py --dataset "D:\FISICA MEDICA\radiomics_eco\mixed"


#physical_devices = tf.config.experimental_list_devices()
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",help="dataset path",required=True)
ap.add_argument("-o","--output",help="path to save the output model",default="nn_ecographic_prediction.h5")

args=vars(ap.parse_args())

EPOCHS=40
BS=128
INIT_LR=1e-2



def load_img(path,target_dim):
    """Load, grayscale and resize the img"""
    img=cv2.imread(path)
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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
            X.append(load_img(i,(299,299)))
            y.append(c)


print(f"[INFO] loading completed len X: {len(X)}\t len y: {len(y)}")


#1. load pre-trained network

baseModel=InceptionV3(include_top=False,
    weights="imagenet", input_shape=(299, 299, 3)
)



#prepare the data


X,y=shuffle(X,y)
lb=LabelBinarizer()

y=np.array(lb.fit_transform(y))
y=to_categorical(y)

X=np.array(X)

print(f"[INFO] extracting image embeddings..")
X=baseModel.predict(X)

X=np.reshape(X,(len(X),X.shape[1]*X.shape[2]*X.shape[3]))
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20)



print(f"[INFO] Check dimension pre-training:\n x_train : {x_train.shape}\t x_test : {x_test.shape}\n y_train : {y_train.shape}\t\t y_test : {y_test.shape}")



### HERE RANDOM FOREST

rf=RandomForestClassifier(n_estimators=500)
rf.fit(x_train,y_train)




print(f"[INFO] evaluating the Random Forest..")

print(f"[SCORE]: RandomForest: {rf.score(x_test,y_test)}")
predIdx=rf.predict(x_test)

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


### SVM

svm_clf=svm.NuSVC(gamma='auto')
svm_clf.fit(x_train,y_train.argmax(axis=1))

print(f"[INFO] evaluating the Support Vector Machine..")

print(f"[SCORE]: Suppor Vector Machine: {svm_clf.score(x_test,y_test.argmax(axis=1))}")
predIdx=svm_clf.predict(x_test)


#predIdx=np.argmax(predIdx,axis=1) #get the max

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


### ENSEMBLE APPROACH ###

print(f"[INFO] Training an ensamble voting classifier..")

lr=LogisticRegression(max_iter=1e3)
svm_p=svm.NuSVC(probability=True)
voting=VotingClassifier(estimators=[('rf',rf),('lr',lr),('svm',svm_p)],voting="soft")

voting.fit(x_train,y_train.argmax(axis=1))
print(f"[SCORE] voting: {voting.score(x_test,y_test.argmax(axis=1))}")

predIdx=voting.predict(x_test)


#predIdx=np.argmax(predIdx,axis=1) #get the max

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



print(f"[INFO] Final fit to save the model..")
filename = os.path.join("models","inception_voting.sav")
pickle.dump(voting, open(filename, 'wb'))

print(f"[INFO] Model saved.")

CROSS=False
if CROSS:

    print(f"[FINAL TEST]: Running Cross Validation on Ensemble Voting Classifier")

    scores = cross_validate(voting, X, y.argmax(axis=1), cv=5)

    print(f"[SCORES]: {scores}")

