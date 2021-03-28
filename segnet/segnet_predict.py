
import tensorflow as tf
from keras_segmentation.predict import predict_multiple
from keras_segmentation.predict import model_from_checkpoint_path

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

print(f"[INFO] Predicting on test set mask..")

pdr=predict_multiple(
	checkpoints_path=r"C:\Users\matte\PycharmProjects\ecographic_breast_nn\segnet\checkpoints\psp_unet",
	inp_dir=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\images_val",
	out_dir=r"C:\Users\matte\PycharmProjects\ecographic_breast_nn\segnet\outputs"
)

model=model_from_checkpoint_path(r"C:\Users\matte\PycharmProjects\ecographic_breast_nn\segnet\checkpoints\psp_unet")


print(f"[INFO] Evaluating model..")
print(model.evaluate_segmentation( inp_images_dir=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\images_val"  , annotations_dir=r"D:\FISICA MEDICA\radiomics_eco\Dataset_BUSI_with_GT\segnet\masks_val" ) )
