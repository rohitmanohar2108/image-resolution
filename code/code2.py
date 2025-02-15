import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2

from google.colab import files

files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets download -d adityachandrasekhar/image-super-resolution

! mkdir image-super-resolution

! unzip image-super-resolution.zip -d image-super-resolution

def process_image(image):
    return image/255

def load_data(path):
    high_res_images = []
    low_res_images = []
    for dirname, _, filenames in os.walk(path+'low_res'):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img = process_image(img)
            low_res_images.append(img)

    for dirname, _, filenames in os.walk(path+'high_res'):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img = process_image(img)
            high_res_images.append(img)

    return np.array(low_res_images), np.array(high_res_images)

base_dir = "/content/image-super-resolution/dataset/"

train_x, train_y =  load_data(base_dir+'train/')
val_x, val_y = load_data(base_dir+'val/')

from google.colab.patches import cv2_imshow

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    high_res_images = []
    low_res_images = []
    for dirname, _, filenames in os.walk(os.path.join(path, 'low_res')):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img = process_image(img)
            low_res_images.append(img)

    for dirname, _, filenames in os.walk(os.path.join(path, 'high_res')):
        for filename in filenames:
            img = cv2.imread(os.path.join(dirname, filename))
            img = process_image(img)
            high_res_images.append(img)

    return np.array(low_res_images), np.array(high_res_images)

base_dir = "/content/image-super-resolution/dataset/"
train_x, train_y = load_data(os.path.join(base_dir, 'train/'))
val_x, val_y = load_data(os.path.join(base_dir, 'val/'))

# Visualize some sample images
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_x[i])
    plt.title('Low Res Image')
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(train_y[i])
    plt.title('High Res Image')
    plt.axis('off')

plt.show()


import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D,concatenate,ZeroPadding2D,UpSampling2D
from keras.optimizers import Adam,SGD

def res(prev_layer):
    l1=Conv2D(64,(3,3),activation='relu')(prev_layer)
    l2=Conv2D(64,(3,3),activation='relu',padding='same')(l1)
    l3=Conv2D(64,(3,3),activation='relu',padding='same')(l2)
    l4=concatenate([l1,l3])
    l4=ZeroPadding2D(padding=(1, 1))(l4)
    l5=Conv2D(64,(3,3),activation='relu',padding='same')(l4)
    l6=concatenate([prev_layer,l5])
    return l6

inp=Input(shape=(256,256,3))
l1=Conv2D(64,(3,3),activation='relu',padding='same')(inp)
l2=res(l1)
l2=UpSampling2D()(l2)
l2=MaxPooling2D()(l2)


l3=res(l2)
l3=UpSampling2D()(l3)
l3=MaxPooling2D()(l3)


l8=res(l3)
l8=UpSampling2D()(l8)
l8=MaxPooling2D()(l8)


l9=res(l8)
l9=UpSampling2D()(l9)
l9=MaxPooling2D()(l9)


l10=res(l9)
l10=UpSampling2D()(l10)
l10=MaxPooling2D()(l10)


l11=res(l10)
l11=UpSampling2D()(l11)
l11=MaxPooling2D()(l11)


l11=concatenate([l10,l11])


l12=res(l11)
l12=UpSampling2D()(l12)
l12=MaxPooling2D()(l12)





l13=res(l12)
l13=UpSampling2D()(l13)
l13=MaxPooling2D()(l13)

l13=concatenate([l8,l13])



l14=res(l13)
l14=UpSampling2D()(l14)
l14=MaxPooling2D()(l14)





l15=res(l14)
l15=UpSampling2D()(l15)
l15=MaxPooling2D()(l15)

l15=concatenate([l2,l15])

out1=Conv2D(3,(3,3),activation='relu')(l15)
out=ZeroPadding2D(padding=(1, 1))(out1)

model=Model(inputs=[inp],outputs=out)
model.summary()

def compute_psnr(original_image, generated_image):

    original_image = tf.convert_to_tensor(original_image, dtype = tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype = tf.float32)

    psnr = tf.image.psnr(original_image, generated_image, max_val = 1.0)

    return tf.math.reduce_mean(psnr, axis = None, keepdims = False, name = None)

def compute_ssim(original_image, generated_image):

    original_image = tf.convert_to_tensor(original_image, dtype = tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype = tf.float32)

    ssim = tf.image.ssim(original_image, generated_image, max_val = 1.0, filter_size = 11, filter_sigma = 1.5, k1 = 0.01, )

    return tf.math.reduce_mean(ssim, axis = None, keepdims = False, name = None)

import math
from keras import backend as K



def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[PSNR,compute_ssim])

# Assuming the training part is as follows
hist = model.fit(train_x, train_y, epochs=1, batch_size=1)

# Now, to save the model
model.save('my_model.h5')  # This will save the architecture, weights, and training configuration

print("Model saved successfully!")

from tensorflow.keras.models import load_model
import tensorflow as tf
import keras.backend as K

# Define the PSNR function as before
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * tf.math.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) / tf.math.log(10.0))

# Define the compute_ssim function
def compute_ssim(original_image, generated_image):

    original_image = tf.convert_to_tensor(original_image, dtype = tf.float32)
    generated_image = tf.convert_to_tensor(generated_image, dtype = tf.float32)

    ssim = tf.image.ssim(original_image, generated_image, max_val = 1.0, filter_size = 11, filter_sigma = 1.5, k1 = 0.01, )

    return tf.math.reduce_mean(ssim, axis = None, keepdims = False, name = None)

# Load the model and specify the custom_objects argument with both custom functions
model_path = '/content/my_model.h5'
model = load_model(model_path, custom_objects={'PSNR': PSNR, 'compute_ssim': compute_ssim})


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

def load_and_process_image(file_path, model, img_size=(256, 256)):
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Load the image
    img = load_img(file_path, target_size=img_size)

    # Convert image to array and normalize
    img_array = img_to_array(img) / 255.0

    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Predict enhanced image using the model
    enhanced_img_array = model.predict(img_array)[0]

    # Clip pixel values to [0, 1]
    enhanced_img_array = np.clip(enhanced_img_array, 0, 1)

    # Convert back to uint8 and scale to [0, 255]
    original_img = (img_array[0] * 255).astype(np.uint8)
    enhanced_img = (enhanced_img_array * 255).astype(np.uint8)

    # Plot original and enhanced images
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(enhanced_img)
    axs[1].set_title('Enhanced Image')
    axs[1].axis('off')
    plt.show()

# Example usage
file_path = '/content/Screenshot 2024-03-18 011542.png'
load_and_process_image(file_path, model)

y_pred=model.predict(train_x[0:1])

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

def load_and_process_image(file_path, model, img_size=(256, 256)):
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Load the image
    img = load_img(file_path, target_size=img_size)

    # Convert image to array and normalize
    img_array = img_to_array(img) / 255.0

    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Predict enhanced image using the model
    enhanced_img_array = model.predict(img_array)[0]

    # Clip pixel values to [0, 1]
    enhanced_img_array = np.clip(enhanced_img_array, 0, 1)

    # Convert back to uint8 and scale to [0, 255]
    original_img = (img_array[0] * 255).astype(np.uint8)
    enhanced_img = (enhanced_img_array * 255).astype(np.uint8)

    # Plot original and enhanced images
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(enhanced_img)
    axs[1].set_title('Enhanced Image')
    axs[1].axis('off')
    plt.show()

# Example usage
file_path = '/content/Screenshot 2024-03-17 235826.png'
load_and_process_image(file_path, model)

