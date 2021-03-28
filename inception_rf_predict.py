import glob

from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pickle
from tensorflow.keras.applications import InceptionV3
import cv2
from sklearn import pipeline
import argparse
import os
from imutils import paths
import pydicom as dcm
from sklearn.pipeline import Pipeline
import tensorflow as tf
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",help="image to predict")
ap.add_argument("-d","--directory",help="normal images to predict")
ap.add_argument("-s","--scan_directory",help="scan subdirectories for dicom files")

args=vars(ap.parse_args())



gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)


def load_img(path,target_dim):
    print(f"[INFO] reading {path}")
    if (".dcm" in path) or ("DCM" in path):
        print(f"Reading as DICOM file")
        ##this is a pydicom file!
        dcm_slice=dcm.dcmread(path)
        img = dcm_slice.pixel_array
        img=cv2.resize(img, target_dim) / 255.
    else:
        print(f"Reading as standard image file")
        img=cv2.imread(path)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img, target_dim) / 255.
    return img





#selezionare modalità immagine oppure
list=[]
if args["directory"] is not None:
    list=paths.list_images(args["directory"])
    img=[load_img(i,(299,299)) for i in list]
    img=np.array(img)

elif args["scan_directory"] is not None:
    base_path=args["scan_directory"]
    print(f"[INFO] scanning {base_path} subdirs for dicom files")
    subdirs=os.listdir(base_path)
    list=[]
    for s in subdirs:
        list.append(glob.glob(os.path.join(base_path,s,"*.DCM"))[0])

    img = [load_img(i, (299, 299)) for i in list]
    img = np.array(img)
else:

    img=load_img(args["image"],(299,299))
    img = np.array(img)
    img = np.expand_dims(img, 0)

#expand dims



def create_model():
    model=InceptionV3(include_top=False,
                            weights="imagenet", input_shape=(299, 299, 3)
                            )
    return model



feature_extractor = create_model()
loaded_model = pickle.load(open('models\inception_voting.sav','rb'))



##prediction


embeddings = feature_extractor.predict(img)
embeddings=np.reshape(embeddings,(len(embeddings),embeddings.shape[1]*embeddings.shape[2]*embeddings.shape[3]))
pred=loaded_model.predict(embeddings)

print(f"[PREDICTION] {pred}")

if len(list):
    for i in range(len(list)):
        print(f"{list[i]}: {pred[i]}")
