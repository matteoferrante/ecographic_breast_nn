import os
from imutils import paths
import cv2
from sklearn.utils import shuffle

"""Data preparation for segnet"""
base_path=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT"

dirs=["benign","malignant"]
imgs=[]
mask=[]

for dir in dirs:
    print(f"[INFO] loading imgs from {dir}")
    im_list=list(paths.list_images(os.path.join(base_path,dir)))
    for (i,img) in enumerate(im_list):
        if "mask" not in img:
            imgs.append(cv2.imread(img))
        else:
            if "mask_" in img:
                x=cv2.imread(img)/255.
                mask[-1]=mask[-1]+x

            else:
                mask.append(cv2.imread(img)/255.)

os.makedirs("segnet",exist_ok=True)

os.makedirs(os.path.join(base_path,"segnet","images_train"),exist_ok=True)
os.makedirs(os.path.join(base_path,"segnet","masks_train"),exist_ok=True)
os.makedirs(os.path.join(base_path,"segnet","images_val"),exist_ok=True)
os.makedirs(os.path.join(base_path,"segnet","masks_val"),exist_ok=True)

cut=0.8
print(type(mask[0]),type(imgs[0]))

imgs,mask=shuffle(imgs,mask)



for (i,(img,m)) in enumerate(zip(imgs,mask)):
    m[m>0.5]=1.
    if i%50==0:
        print(f"[INFO] writing {i}/{len(imgs)}")
    if i<cut*len(imgs):

        cv2.imwrite(os.path.join(base_path,"segnet","images_train",f"{i}.png"),img=img)
        cv2.imwrite(os.path.join(base_path, "segnet", "masks_train",f"{i}.png"),img=m)
    else:
        cv2.imwrite(os.path.join(base_path,"segnet","images_val",f"{i}.png"),img=img)
        cv2.imwrite(os.path.join(base_path, "segnet", "masks_val",f"{i}.png"),img=m)

print(f"[END] imgs: {len(imgs)}\t mask: {len(mask)}")


