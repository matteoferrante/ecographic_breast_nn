import numpy as np
import argparse
from imutils import paths
import cv2
from radiomics import featureextractor
import os
import SimpleITK as sitk

#usage python radiomics_extraction.py --dataset "D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT"

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",help="dataset path",required=True)
ap.add_argument("-o","--output",help="path to save the output model",default="nn_ecographic_prediction.h5")

args=vars(ap.parse_args())


def load_img(path,target_dim):
    """Load, grayscale and resize the img"""
    image=sitk.ReadImage(path)
    image3d=sitk.JoinSeries(image)
    return image3d

def load_mask(path):
    mask=sitk.ReadImage(path)
    mask3d = sitk.JoinSeries(mask)
    return mask3d

print(f"[INFO] Loading dataset..")

classes=os.listdir(args["dataset"])

X=[]
X_path=[]
mask=[]
mask_path=[]
y=[]
for c in classes:
    im_list=paths.list_images(os.path.join(args["dataset"],c))
    for i in im_list:
        if "mask" not in i:
            X.append(load_img(i,(500,500)))
            X_path.append(i)
            y.append(c)
        else:
            if "mask_" not in i:
                mask_path.append(i)
                mask.append(load_mask(i))





print(f"[INFO] loading completed len X: {len(X)}\tlen mask: {len(mask)}\t len y: {len(y)}")

settings = {'label': 255}
extractor = featureextractor.RadiomicsFeatureExtractor(additionalInfo=True,**settings)

result=[]

for (n,(i,l)) in enumerate(zip(X,mask)):
    print(f"[INFO] loading {n}")
    result.append(extractor.execute(i,l))