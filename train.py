# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XlbFByT-f0jToBPWBdQ_0GdUB2pZhHKL
"""

#TRAINING THE MODEL USING VGG16 MODEL (PRE-TRAINED ON 'IMAGENET' DATASET) AND SAVING IT
# Importing the required libraries
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

# Specifying the size of the images applied to the input layer
IMAGE_SIZE = [224, 224]

# Paths to access the training and test dataset
train_path = '/content/drive/MyDrive/Datasets/Train'
valid_path = '/content/drive/MyDrive/Datasets/Test'

# Initialising the VGG16 model object (pre-trained model) - performs transfer learning by importing the weights calculated using 'ImageNet' dataset
# Here, 'include_top = False' parameter neglects the input and the output layers of the vgg16 model in order to customise the input and output categories
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Avoid training existing weights
for layer in vgg.layers:
  layer.trainable = False

# Stores all the pathnames matching the pattern specified
folders = glob('/content/drive/MyDrive/Datasets/Train/*')
  
# Transforming 2D feature-maps into a 1D vector supplied as an input to fully-connected layers
x = Flatten()(vgg.output)

# Adding dense layer to determine the output
# Here, activation function = 'softmax', used for multinomial probability distribution will output values between 0 and 1 for each category,
# where input image is assigned the class with maximum probability.
prediction = Dense(len(folders), activation='softmax')(x) 

# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
print(f"The summary of the model :\n {model.summary()}")

# Compiling the model thereby specifying the loss function and the optimization function to be used.
# Loss function = 'categorical cross-entropy' used since there are multiple label values
# Optimiser = 'adam' used to reach to the global minima while training the model (Stuck in local minima adam helps to get out of local minima and reach global value)
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Performing Data Augmentation - Artificially generating more data so that the model can generalize better
from keras.preprocessing.image import ImageDataGenerator

# Here, 'rescale' - multiplies the rgb pixel values by 1/255 to scale them between 0 and 1
# 'shear_range' - specifies range by which we can perform shearing transformation on images
# 'zoom_range' - specifies range for zooming in images
# 'horizontal_flip' - flips half of the image randomly along the horizontal axis
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

valid_datagen = ImageDataGenerator(rescale = 1./255)

# Extract the images from the ImageGenerator object in batches for training and validating
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 15,
                                                 class_mode = 'categorical')

valid_set = valid_datagen.flow_from_directory('/content/drive/MyDrive/Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 5,
                                            class_mode = 'categorical')

# Fit the model using 'fit_generator' method on the training and validation dataset
r = model.fit_generator(
  training_set,
  validation_data=valid_set,
  epochs=4,
  steps_per_epoch=15,
  validation_steps=5
)

# Plot the loss function values
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# Plot the accuracy values
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# Save the model trained using vgg16(convolutional network) to access it later
model.save('facefeatures_new_model')