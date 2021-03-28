"""Code to train a segmentation net using dataset separated into img and labels"""

from keras_segmentation.predict import predict_multiple
from keras_segmentation.models.unet import vgg_unet
import tensorflow as tf
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_voc12
from keras_segmentation.models.pspnet import pspnet_50
from keras_segmentation.models.unet import unet_mini

from keras_segmentation.predict import evaluate
import gc
gc.collect()

#rifare training con data augmentation, provare altre reti, transfer learning, fare valutazione

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

print(f"[INFO] Building the model..")
#model = vgg_unet(n_classes=2 ,  input_height=512, input_width=512  )


pretrained_model =  pspnet_50_ADE_20K()

#model = pspnet_50( n_classes=2 ) #new model

model = pspnet_50( n_classes=2 ) #new model

transfer_weights( pretrained_model , model  ) # transfer weights from pre-trained model to your model


print(f"[INFO] Training the model..")
model.train(
    train_images =  r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\images_train",
    train_annotations = r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\masks_train",
    checkpoints_path =r"C:\Users\matte\PycharmProjects\ecographic_breast_nn\segnet\checkpoints\psp_unet", epochs=5,batch_size=1)



print(f"[INFO] Running predictions")
pdr=model.predict_multiple(
	inp_dir=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\images_val",
	out_dir=r"C:\Users\matte\PycharmProjects\ecographic_breast_nn\segnet\outputs\overlay",overlay_img=True
)

print(f"[INFO] Evaluating model..")
print(model.evaluate_segmentation( inp_images_dir=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\images_val"  , annotations_dir="D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\masks_val" ) )