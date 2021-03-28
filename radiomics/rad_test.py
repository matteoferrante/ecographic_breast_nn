import os

import SimpleITK as sitk
import nrrd
import numpy as np
from radiomics import firstorder
from radiomics.featureextractor import RadiomicsFeatureExtractor
import cv2
from imutils import paths
import pandas as pd
##loading data
base_path=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT"
classes=["benign","malignant"]

def load_img(imageName):
    img = sitk.ReadImage(imageName, sitk.sitkInt8)
    if len(img.GetSize()) == 2:
        img = sitk.GetArrayFromImage(img)
        img = np.expand_dims(img, 0)
        img = sitk.GetImageFromArray(img)
    return img

def load_mask(maskName):
    mask = sitk.ReadImage(maskName)

    if len(mask.GetSize()) == 2:
        mask = sitk.GetArrayFromImage(mask)
        mask = np.expand_dims(mask, 0)
        mask = sitk.GetImageFromArray(mask)
    return mask



X=[]
masks=[]
y=[]

for c in classes:
    im_list=paths.list_images(os.path.join(base_path,c))
    for i in im_list:
        if "mask" not in i:
            X.append(load_img(i))
            y.append(c)
        else:
            if "mask_" not in i:
                masks.append(load_mask(i))




settings = {'interpolator': sitk.sitkBSpline,'label': 255}
extractor = RadiomicsFeatureExtractor(**settings)
extractor.enableAllImageTypes()


results=[]
for i in range(len(X)):
    if i%50==0:
        print(f"[INFO] Evaluating {i+1}/{len(X)}")
    d=dict(extractor.execute(X[i],masks[i]))
    status={"status":y[i]}
    d.update(status)
    results.append(d)

print(f"[INFO] Saving results to radiomics_features_extracted.csv")

df=pd.DataFrame.from_dict(results)
df.to_csv(r"radiomics_features_extracted.csv")





