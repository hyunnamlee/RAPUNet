import tensorflow as tf
import albumentations as albu
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score
from ModelArchitecture.DiceLoss import dice_metric_loss
from ModelArchitecture import RAPUNet
from ImageLoader import ImageLoader2D
import tensorflow_addons as tfa 
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops
    

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_size = 352
dataset_type = 'CVC-ColonDB' # Options: kvasir/cvc-clinicdb/cvc-colondb/etis-laribpolypdb
learning_rate = 1e-4
seed_value = 58800
filters = 17 

starter_learning_rate = 1e-4
end_learning_rate = 1e-6
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.2)

opts = tfa.optimizers.AdamW(learning_rate = 1e-4, weight_decay = learning_rate_fn)
b_size = 2

ct = datetime.now()

model_type = "RAPUNet"

progress_path = 'ProgressFull/' + dataset_type + '_progress_csv_' + model_type + '_filters_' + str(filters) +  '_' + str(ct) + '.csv'
progressfull_path = 'ProgressFull/' + dataset_type + '_progress_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.txt'
plot_path = 'ProgressFull/' + dataset_type + '_progress_plot_' + model_type + '_filters_' + str(filters) + '_' + str(ct) + '.png'
model_path = 'ModelSaveTensorFlow/' + dataset_type + '/' + model_type + '_filters_' + str(filters) + '_' + str(ct)

EPOCHS = 300
min_loss_for_saving = 0.1

model = RAPUNet.create_model(img_height=img_size, img_width=img_size, input_chanels=3, out_classes=1, starting_filters=filters)  
model.compile(optimizer=opts, loss=dice_metric_loss) #BCE, metric, optimizer, opts

model.summary()
print(get_flops(model))

data_path = "../data/Train_Clinic/"  # Add the path to your train data directory
test_path = "../data/TestDataset/CVC-ClinicDB/" # Add the path to your test data directory
X, Y = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', data_path)

# use data split train/valid/test as 0.8/0.1/0.1
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle= True, random_state = seed_value)
#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.111, shuffle= True, random_state = seed_value)

# use fixed test data
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, shuffle= True, random_state = seed_value)
x_test, y_test = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', test_path)


aug_train = albu.Compose([
    albu.HorizontalFlip(),
    albu.VerticalFlip(),
    albu.ColorJitter(brightness=(0.6,1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
    albu.Affine(scale=(0.5,1.5), translate_percent=(-0.125,0.125), rotate=(-180,180), shear=(-22.5,22), always_apply=True),
])

def augment_images():
    x_train_out = []
    y_train_out = []

    for i in range (len(x_train)):
        ug = aug_train(image=x_train[i], mask=y_train[i])
        x_train_out.append(ug['image'])  
        y_train_out.append(ug['mask'])

    return np.array(x_train_out), np.array(y_train_out)
    

step = 0

for epoch in range(0, EPOCHS):
    
    print(f'Training, epoch {epoch}')
    print('Learning Rate: ' + str(learning_rate))
    
    step += 1
        
    image_augmented, mask_augmented = augment_images()
    
    csv_logger = CSVLogger(progress_path, append=True, separator=';')
    
    model.fit(x=image_augmented, y=mask_augmented, epochs=1, batch_size=b_size, validation_data=(x_valid, y_valid), verbose=1, callbacks=[csv_logger])
    
    prediction_valid = model.predict(x_valid, verbose=0)
    loss_valid = dice_metric_loss(y_valid, prediction_valid)
    loss_valid = loss_valid.numpy()
    print("Loss Validation: " + str(loss_valid))
        
    prediction_test = model.predict(x_test, verbose=0)
    loss_test = dice_metric_loss(y_test, prediction_test)
    loss_test = loss_test.numpy()
    print("Loss Test: " + str(loss_test))
        
    with open(progressfull_path, 'a') as f:
        f.write('epoch: ' + str(epoch) + '\nval_loss: ' + str(loss_valid) + '\ntest_loss: ' + str(loss_test) + '\n\n\n')
        #f.write('epoch: ' + str(epoch) + '\nval_loss: ' + str(loss_valid) + '\n\n\n')
    

    if min_loss_for_saving > loss_valid:
        min_loss_for_saving = loss_valid
        print("Saved model with val_loss: ", loss_valid)
        model.save(model_path)
        model.save("best_model.h5")        
 
    del image_augmented
    del mask_augmented

    gc.collect()

   
print("Loading the model")

kpath = "../data/TestDataset/Kvasir/" # Add the path to your data directory Kvasir CVC-ClinicDB Kvasir/
x_kvasir, y_kvasir = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', kpath)

cnpath = "../data/TestDataset/CVC-ClinicDB/" # Add the path to your data directory Kvasir CVC-ClinicDB
x_clinic, y_clinic = ImageLoader2D.load_data(img_size, img_size, -1, 'kvasir', cnpath)

model = tf.keras.models.load_model(model_path, custom_objects={'dice_metric_loss':dice_metric_loss}) 

prediction_train = model.predict(x_train, batch_size=1)
prediction_valid = model.predict(x_valid, batch_size=1)
prediction_test = model.predict(x_test, batch_size=1)

prediction_kvasir = model.predict(x_kvasir, batch_size=b_size)
prediction_clinic = model.predict(x_clinic, batch_size=b_size)
 
print("Predictions done")

dice_train = f1_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                           np.ndarray.flatten(prediction_train > 0.5))
dice_test = f1_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                          np.ndarray.flatten(prediction_test > 0.5))
dice_valid = f1_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                           np.ndarray.flatten(prediction_valid > 0.5))
dice_kvasir = f1_score(np.ndarray.flatten(np.array(y_kvasir, dtype=bool)),
                           np.ndarray.flatten(prediction_kvasir > 0.5)) 
dice_clinic = f1_score(np.ndarray.flatten(np.array(y_clinic, dtype=bool)),
                           np.ndarray.flatten(prediction_clinic > 0.5))
                          
print("Dice finished")


miou_train = jaccard_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                           np.ndarray.flatten(prediction_train > 0.5))
miou_test = jaccard_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                          np.ndarray.flatten(prediction_test > 0.5))
miou_valid = jaccard_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                           np.ndarray.flatten(prediction_valid > 0.5))

print("Miou finished")


precision_train = precision_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                  np.ndarray.flatten(prediction_train > 0.5))
precision_test = precision_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                                 np.ndarray.flatten(prediction_test > 0.5))
precision_valid = precision_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                  np.ndarray.flatten(prediction_valid > 0.5))

print("Precision finished")


recall_train = recall_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                            np.ndarray.flatten(prediction_train > 0.5))
recall_test = recall_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                           np.ndarray.flatten(prediction_test > 0.5))
recall_valid = recall_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                            np.ndarray.flatten(prediction_valid > 0.5))

print("Recall finished")


accuracy_train = accuracy_score(np.ndarray.flatten(np.array(y_train, dtype=bool)),
                                np.ndarray.flatten(prediction_train > 0.5))
accuracy_test = accuracy_score(np.ndarray.flatten(np.array(y_test, dtype=bool)),
                               np.ndarray.flatten(prediction_test > 0.5))
accuracy_valid = accuracy_score(np.ndarray.flatten(np.array(y_valid, dtype=bool)),
                                np.ndarray.flatten(prediction_valid > 0.5))


print("Accuracy finished")

final_file = 'results_' + model_type + '_' + str(filters) + '_' + dataset_type + 'CAformer_clean_et.txt'
print(final_file)


with open(final_file, 'a') as f:
    f.write(dataset_type + '\n\n')
    f.write(model_path + '\n\n')
    f.write('dice_kvasir: ' + str(dice_kvasir)+ ' dice_clinic: ' + str(dice_clinic)+' dice_train: ' + str(dice_train) + ' dice_valid: ' + str(dice_valid) + ' dice_test: ' + str(dice_test) + '\n\n')
    
    f.write('miou_train: ' + str(miou_train) + ' miou_valid: ' + str(miou_valid) + ' miou_test: ' + str(miou_test) + '\n\n')
    f.write('precision_train: ' + str(precision_train) + ' precision_valid: ' + str(precision_valid) + ' precision_test: ' + str(precision_test) + '\n\n')
    f.write('recall_train: ' + str(recall_train) + ' recall_valid: ' + str(recall_valid) + ' recall_test: ' + str(recall_test) + '\n\n')
    f.write('accuracy_train: ' + str(accuracy_train) + ' accuracy_valid: ' + str(accuracy_valid) + ' accuracy_test: ' + str(accuracy_test) + '\n\n\n\n')
'''    

with open(final_file, 'a') as f:
    f.write(dataset_type + '\n\n')
    f.write(model_path + '\n\n')
    f.write('dice_kvasir: ' + str(dice_kvasir)+ '  dice_clinic: ' + str(dice_clinic)+' dice_train: ' + str(dice_train) + ' dice_valid: ' + str(dice_valid)  + '\n\n')
    
    f.write('miou_train: ' + str(miou_train) + ' miou_valid: ' + str(miou_valid)  + '\n\n')
    f.write('precision_train: ' + str(precision_train) + ' precision_valid: ' + str(precision_valid) + '\n\n')
    f.write('recall_train: ' + str(recall_train) + ' recall_valid: ' + str(recall_valid) + '\n\n')
    f.write('accuracy_train: ' + str(accuracy_train) + ' accuracy_valid: ' + str(accuracy_valid) + '\n\n\n\n')    
    
print('File done')    
'''    
    
    
