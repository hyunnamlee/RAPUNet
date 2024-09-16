import tensorflow as tf
import albumentations as albu
import numpy as np

from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import RAPUNet
from CustomLayers import ImageLoader2D
import tensorflow_addons as tfa 

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_size = 352

model_type = "RAPUNet"
   
print("Loading the model and testdata")

#model_p = 'RAPUNet_Kvasir.h5'
#model_p = 'RAPUNet_ClinicDB.h5'
model_p = 'RAPUNet_kc.h5'

model = tf.keras.models.load_model(model_p, custom_objects={'dice_metric_loss':dice_metric_loss}) 

kpath = "../data/TestDataset/Kvasir/" # Add the path to your test data directory 
x_kvasir, y_kvasir = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', kpath)

cnpath = "../data/TestDataset/CVC-ClinicDB/" # Add the path to your test data directory 
x_clinic, y_clinic = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', cnpath)

epath = "../data/TestDataset/ETIS-LaribPolypDB/" # Add the path to your test data directory
x_et, y_et = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', epath)

ccpath = "../data/TestDataset/CVC-ColonDB/" # Add the path to your data test directory 
x_cc, y_cc = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', ccpath)

c3path = "../data/TestDataset/CVC-300/" # Add the path to your data test directory 
x_c3, y_c3 = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', c3path)
 

prediction_kvasir = model.predict(x_kvasir, batch_size=1)
prediction_clinic = model.predict(x_clinic, batch_size=1)
prediction_cc = model.predict(x_cc, batch_size=1)
prediction_et = model.predict(x_et, batch_size=1)
prediction_c3 = model.predict(x_c3, batch_size=1)
 
print("Predictions done")

dice_kvasir = f1_score(np.ndarray.flatten(np.array(y_kvasir, dtype=bool)),
                           np.ndarray.flatten(prediction_kvasir > 0.5)) 
dice_clinic = f1_score(np.ndarray.flatten(np.array(y_clinic, dtype=bool)),
                           np.ndarray.flatten(prediction_clinic > 0.5))
dice_cc = f1_score(np.ndarray.flatten(np.array(y_cc, dtype=bool)),
                           np.ndarray.flatten(prediction_cc > 0.5))                           
dice_c3 = f1_score(np.ndarray.flatten(np.array(y_c3, dtype=bool)),
                           np.ndarray.flatten(prediction_c3 > 0.5))                          
dice_et = f1_score(np.ndarray.flatten(np.array(y_et, dtype=bool)),
                           np.ndarray.flatten(prediction_et > 0.5))
                           
print("Dice finished")


miou_kvasir = jaccard_score(np.ndarray.flatten(np.array(y_kvasir, dtype=bool)),
                           np.ndarray.flatten(prediction_kvasir > 0.5))
miou_clinic = jaccard_score(np.ndarray.flatten(np.array(y_clinic, dtype=bool)),
                          np.ndarray.flatten(prediction_clinic > 0.5))
miou_cc = jaccard_score(np.ndarray.flatten(np.array(y_cc, dtype=bool)),
                           np.ndarray.flatten(prediction_cc > 0.5))
miou_c3 = jaccard_score(np.ndarray.flatten(np.array(y_c3, dtype=bool)),
                           np.ndarray.flatten(prediction_c3 > 0.5))
miou_et = jaccard_score(np.ndarray.flatten(np.array(y_et, dtype=bool)),
                          np.ndarray.flatten(prediction_et > 0.5))

print("Miou finished")


precision_kvasir = precision_score(np.ndarray.flatten(np.array(y_kvasir, dtype=bool)),
                                  np.ndarray.flatten(prediction_kvasir > 0.5))
precision_clinic = precision_score(np.ndarray.flatten(np.array(y_clinic, dtype=bool)),
                                 np.ndarray.flatten(prediction_clinic > 0.5))
precision_cc = precision_score(np.ndarray.flatten(np.array(y_cc, dtype=bool)),
                                  np.ndarray.flatten(prediction_cc > 0.5))
precision_c3 = precision_score(np.ndarray.flatten(np.array(y_c3, dtype=bool)),
                                  np.ndarray.flatten(prediction_c3 > 0.5))
precision_et = precision_score(np.ndarray.flatten(np.array(y_et, dtype=bool)),
                                 np.ndarray.flatten(prediction_et > 0.5))
                              

print("Precision finished")


recall_kvasir = recall_score(np.ndarray.flatten(np.array(y_kvasir, dtype=bool)),
                            np.ndarray.flatten(prediction_kvasir > 0.5))
recall_clinic = recall_score(np.ndarray.flatten(np.array(y_clinic, dtype=bool)),
                           np.ndarray.flatten(prediction_clinic > 0.5))
recall_cc = recall_score(np.ndarray.flatten(np.array(y_cc, dtype=bool)),
                            np.ndarray.flatten(prediction_cc > 0.5))
recall_c3 = recall_score(np.ndarray.flatten(np.array(y_c3, dtype=bool)),
                            np.ndarray.flatten(prediction_c3 > 0.5))
recall_et = recall_score(np.ndarray.flatten(np.array(y_et, dtype=bool)),
                           np.ndarray.flatten(prediction_et > 0.5))

print("Recall finished")


accuracy_kvasir = accuracy_score(np.ndarray.flatten(np.array(y_kvasir, dtype=bool)),
                                np.ndarray.flatten(prediction_kvasir > 0.5))
accuracy_clinic = accuracy_score(np.ndarray.flatten(np.array(y_clinic, dtype=bool)),
                               np.ndarray.flatten(prediction_clinic > 0.5))
accuracy_cc = accuracy_score(np.ndarray.flatten(np.array(y_cc, dtype=bool)),
                                np.ndarray.flatten(prediction_cc > 0.5))        
accuracy_et = accuracy_score(np.ndarray.flatten(np.array(y_et, dtype=bool)),
                               np.ndarray.flatten(prediction_et > 0.5))
accuracy_c3 = accuracy_score(np.ndarray.flatten(np.array(y_c3, dtype=bool)),
                                np.ndarray.flatten(prediction_c3 > 0.5))

print("Accuracy finished")

final_file = 'results_final_performance.txt'
print(final_file)

with open(final_file, 'a') as f:
    f.write(model_p + '\n\n')    
    f.write('dice_kvasir: ' + str(dice_kvasir)+ '   dice_clinic: ' + str(dice_clinic)+'  dice_et: ' + str(dice_et) +' dice_cc: ' + str(dice_cc) + ' dice_c3: ' + str(dice_c3)  + '\n\n')
    f.write('miou_kvasir: ' + str(miou_kvasir)+ '   miou_clinic: ' + str(miou_clinic)+'  miou_et: ' + str(miou_et) +' miou_cc: ' + str(miou_cc) + ' miou_c3: ' + str(miou_c3)  + '\n\n')
    f.write('precision_kvasir: ' + str(precision_kvasir)+ '   precision_clinic: ' + str(precision_clinic)+'  precision_et: ' + str(precision_et) +' precision_cc: ' + str(precision_cc) + ' precision_c3: ' + str(precision_c3)  + '\n\n')
    f.write('recall_kvasir: ' + str(recall_kvasir)+ '   recall_clinic: ' + str(recall_clinic)+'  recall_et: ' + str(recall_et) +' recall_cc: ' + str(recall_cc) + ' recall_c3: ' + str(recall_c3)  + '\n\n')
    f.write('accuracy_kvasir: ' + str(accuracy_kvasir)+ '   accuracy_clinic: ' + str(accuracy_clinic)+'  accuracy_et: ' + str(accuracy_et) +' accuracy_cc: ' + str(accuracy_cc) + ' accuracy_c3: ' + str(accuracy_c3)  + '\n\n')
    
print('File done')    
    
    
    
