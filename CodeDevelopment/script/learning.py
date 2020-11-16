# Librairies
print("Load Libraries")
import time
import pickle
import os
import tensorflow.keras.preprocessing.image as kpi
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km

from tensorflow.python.client import device_lib

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Argument
import argparse

# TODO Write here the parameters that can be given as inputs to the algorithm.
parser = argparse.ArgumentParser()
DATA_PATH = '/Users/cecile/Documents/INSA/5A/AI-Frameworks/CodeDevelopment'

parser.add_argument('--data_dir', type=str, default=DATA_PATH+'/data/')
parser.add_argument('--results_dir', type=str, default=DATA_PATH+'/results/')
parser.add_argument('--model-dir', type=str, default=DATA_PATH+'/model/')

parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
args = parser.parse_args()


## Data Generator

img_width = 150
img_height = 150

N_train = len(os.listdir(args.data_dir+"/train/cats/")) + len(os.listdir(args.data_dir+"/train/dogs/"))
N_val = len(os.listdir(args.data_dir+"/validation/cats/")) + len(os.listdir(args.data_dir+"/validation/dogs/"))
print("%d   %d"%(N_train, N_val))

# TODO Write here code to generate obh train and validation generator
train_datagen = kpi.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    args.data_dir + "/train/",  # this is the target directory
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='binary')

valid_datagen = kpi.ImageDataGenerator(rescale=1. / 255)

validation_generator = valid_datagen.flow_from_directory(
    args.data_dir + "/validation/",
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='binary')


## Model
# TODO Write a simple convolutional neural network
model_conv = km.Sequential()
model_conv.add(kl.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), data_format="channels_last"))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Conv2D(32, (3, 3)))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Conv2D(64, (3, 3)))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.MaxPooling2D(pool_size=(2, 2)))

model_conv.add(kl.Flatten())  # this converts our 3D feature maps to 1D feature vectors
model_conv.add(kl.Dense(64))
model_conv.add(kl.Activation('relu'))
model_conv.add(kl.Dropout(0.5))
model_conv.add(kl.Dense(1))
model_conv.add(kl.Activation('sigmoid'))

model_conv.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])



## Learning

print("Start Learning")
ts = time.time()
history = model_conv.fit_generator(train_generator, steps_per_epoch=N_train // args.batch_size, epochs=args.epochs,
                         validation_data=validation_generator, validation_steps=N_val // args.batch_size)
te = time.time()
t_learning = te - ts


## Test

print("Start predicting")
ts = time.time()
score_train = model_conv.evaluate_generator(train_generator, N_train / args.batch_size, verbose=1)
score_val = model_conv.evaluate_generator(validation_generator, N_val / args.batch_size, verbose=1)
te = time.time()
t_prediction = te - ts


## Save Model

# TODO Save model in model folder
# Pour avoir un nom generique
args_str = "epochs_%d_batch_size_%d" %(args.epochs, args.batch_size)
model_conv.save(args.model_dir + "/" + args_str + ".h5")


## Save results
## TODO Save results (learning time, prediction time, train and test accuracy, history.history object) in result folder
print("Save results")
results = vars(args)  #instancie un dict avec les args
results.update({"t_learning": t_learning, "t_prediction": t_prediction, "accuracy_train": score_train,
                 "accuracy_val": score_val, "history" : history.history})
pickle.dump(results, open(args.results_dir + "/" + args_str + ".pkl", "wb"))



















