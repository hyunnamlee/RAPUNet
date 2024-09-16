import tensorflow as tf
import albumentations as albu
import numpy as np
import gc

from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import RAPUNet
from CustomLayers import ImageLoader2D
import tensorflow_addons as tfa 
import cv2
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_size = 352
   
print("Loading the model and testdata")

kpath = "../data/TestDataset/Kvasir/" # Add the path to your test data directory 
x_kvasir, y_kvasir = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', kpath)

#cnpath = "../data/TestDataset/CVC-ClinicDB/" # Add the path to your test data directory 
#x_clinic, y_clinic = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', cnpath)

model_p = 'RAPUNet_Kvasir.h5'
#model_p = 'RAPUNet_ClinicDB.h5'
#model_p = 'RAPUNet_kc.h5'

model = tf.keras.models.load_model(model_p, custom_objects={'dice_metric_loss':dice_metric_loss}) 
  
prediction_kvasir = model.predict(x_kvasir, batch_size=1)
#prediction_clinic = model.predict(x_clinic, batch_size=1)

dataset_type = 'kvasir' # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb
out_path = './predict_img/'
out_dir = os.path.join(out_path, dataset_type)
out_img = os.path.join(out_dir, 'img')
out_mask = os.path.join(out_dir, 'mask')
out_result = os.path.join(out_dir, 'result')

if not os.path.exists(out_img):
    os.makedirs(out_img)
    os.makedirs(out_mask)
    os.makedirs(out_result)

index=0;    
for item in prediction_kvasir: #prediction_ clinic: kvasir:

    temp = np.squeeze(item)*255
    img_out = out_result+'/result_'+str(index) +'.png'
    cv2.imwrite(img_out, temp)
    index= index+1

index=0    
for item in x_kvasir: #x_kvasir:

    temp = np.squeeze(item)*255
    img_out = out_img+'/img_'+str(index) +'.png'
    cv2.imwrite(img_out, temp)
    index= index+1

index=0    
for item in y_kvasir: #y_kvasir:

    temp = np.squeeze(item)*255
    img_out = out_mask+'/mask_'+str(index) +'.png'
    cv2.imwrite(img_out, temp)
    index= index+1
    

   
    
    
