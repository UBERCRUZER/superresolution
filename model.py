# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 19:36:19 2019

@author: UBERCRUZER
"""

from sklearn.datasets import fetch_lfw_people
import pickle
import os
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from PIL import Image

from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add

plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
import keras.backend as K
import skimage.transform
#from skimage import data, io, filters
from numpy import array
from skimage.transform import rescale, resize
#from scipy.misc import imresize

#%%

#%matplotlib inline

#point path to location of saved files (or where you want to save them)
file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--')

##----------------------- UNCOMMENT TO DOWNLOAD DATASET -----------------------
#
#lfw_people = fetch_lfw_people(color=True, resize=1)
##                              
###write download
#outputFile = os.path.join(file_dir, 'allFacesFullResColor.obj')
#pickle.dump(lfw_people, open(outputFile,"wb"))
## ---------WARNING this file is 3.5gb when saved and pickled----------
#
##-----------------------------------------------------------------------------

#read faces
inputfile = os.path.join(file_dir, 'allFacesFullResColor.obj')
lfw_people = pickle.load(open(inputfile,"rb"))

#TEST IMAGE
#plt.imshow(lfw_people.images[4545, :124,1:93].astype(np.uint8))


#%%

# helper functions for saving space

def conv_block(model, k_size, filterss, strides):
    gen = model
    model = Conv2D(filters = filterss, kernel_size = k_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filterss, kernel_size = k_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = add([gen, model])
    return model
    
def up_sampling_block(model, k_size, filterss, strides):
    model = Conv2D(filters = filterss, kernel_size = k_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    return model

def discriminator_block(model, filterss, kernel_size, strides):
    model = Conv2D(filters = filterss, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    return model


class Generator(object):

    def __init__(self, noise_shape):
        self.noise_shape = noise_shape

    def generator(self):
	    gen_input = Input(shape = self.noise_shape)
	    
	    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
	    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
	   
	    gen_model = model
        
        # residual blocks for easier training
	    for index in range(8):  
	        model = conv_block(model, 3, 64, 1)
	    
	    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
	    model = BatchNormalization(momentum = 0.5)(model)
	    model = add([gen_model, model])
	    
	    for index in range(2):
	        model = up_sampling_block(model, 3, 256, 1)
	    
	    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
	    model = Activation('tanh')(model)
	   
	    generator_model = Model(inputs = gen_input, outputs = model)
        
	    return generator_model

class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape
    
    def discriminator(self):
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        model = Flatten()(model)
        
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        return discriminator_model
    
#%%
        
# vgg loss function for "perceptual loss" 
def vgg_loss(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

# for training generator on VGG loss and cross entropy
def get_gan_network(discriminator, shape, generator, optimizer):
    
    # lock generator
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    # generate SR
    generator_output = generator(gan_input)
    # hopefully fool discriminator
    gan_output = discriminator(generator_output)
    # calculate loss and train generator
    gan = Model(inputs=gan_input, outputs=[generator_output,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    return gan


#%%

# preprocessing functions

def hr_images(images):
    images_hr = array(images)
    return images_hr

def lr_images(imagesIn, ds=4):
    # for generating LR
    images = []
    for img in range(len(imagesIn)):
        images.append(skimage.transform.resize(imagesIn[img], [imagesIn[img].shape[0] // ds,
                                                        imagesIn[img].shape[1] // ds], anti_aliasing=False))
    images_lr = array(images)
    return images_lr

def upsample(imagesIn, factor=4):
    # for generating bycubic interpolation
    images = []
    for img in range(len(imagesIn)):
        images.append(skimage.transform.resize(imagesIn[img], [imagesIn[img].shape[0] * factor, 
                                                        imagesIn[img].shape[1] * factor], anti_aliasing=True))
    return array(images)

def normalize(input_data):
    # normalize RGB values
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    # denormalize for display
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8) 


#%%

# paths for image saves
file_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--', 'output')
imgSav = os.path.join(file_dir, 'gan_generated_image_epoch_%d.png')

file_dirMe = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--', 'output', 'me')
imgMeSav = os.path.join(file_dirMe, 'me_generated_image_epoch_%d.png')

def imgOut(epoch, generator):
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=1)
    image_batch_hr = denormalize(x_test_hr[rand_nums])
    image_batch_lr = x_test_lr[rand_nums]
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    image_batch_bc = upsample(image_batch_lr, 4)
        
    plt.figure(figsize=(15, 5))
    
    dim=(1, 4)  # 4 wide
    
    # lowres
    plt.subplot(dim[0], dim[1], 1)
    plt.title('Low Res')
    plt.imshow(image_batch_lr[0], interpolation='nearest')
    plt.axis('off')
    
    # naive resize
    plt.subplot(dim[0], dim[1], 2)
    plt.title('Naive Resize')
    plt.imshow(image_batch_bc[0], interpolation='nearest')
    plt.axis('off')
    
    # generated
    plt.subplot(dim[0], dim[1], 3)
    plt.title('Generated SR')
    plt.imshow(generated_image[0], interpolation='nearest')
    plt.axis('off')
    
    # truth
    plt.subplot(dim[0], dim[1], 4)
    plt.title('Truth')
    plt.imshow(image_batch_hr[0], interpolation='nearest')
    plt.axis('off')
    
    # save file
    plt.tight_layout()
    plt.savefig(imgSav % epoch)
    plt.close('all')
    
    # learning gif images    
    me_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--')
    me_path = os.path.join(me_dir, 'lowresme.jpg')
    me = Image.open(me_path)
    me = np.expand_dims(me, axis=0)
    me_lr = np.array(me)
    me_lr_norm = normalize(me_lr)
    gen_img = generator.predict(me_lr_norm)
    generated_image = denormalize(gen_img)
    plt.imshow(generated_image[0])
    plt.savefig(imgMeSav % epoch)
    plt.close('all')

    
#%%
    
files = lfw_people.images[0:6001, :124,1:93]

# 700 image training set
x_train = files[4300:5000]

# test set
x_test = files[5001:6000]


#%%

np.random.seed(555)

# resizing images 124x92
image_shape = lfw_people.images[0, :124,1:93].shape
channels = image_shape[2]

# path for saving models
file_dir_saves = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--', 'saves')
genSav = os.path.join(file_dir_saves, 'gen_model%d.h5')
discSav = os.path.join(file_dir_saves, 'dis_model%d.h5')
ganSav = os.path.join(file_dir_saves, 'gan_model%d.h5')

epochs = 20000
batch_size = 75

downscale_factor = 4

# image preprocessing
x_train_hr = hr_images(x_train)
x_train_hr = normalize(x_train_hr)

x_train_lr = lr_images(x_train, downscale_factor)
x_train_lr = normalize(x_train_lr)

x_test_hr = hr_images(x_test)
x_test_hr = normalize(x_test_hr)

x_test_lr = lr_images(x_test, downscale_factor)
x_test_lr = normalize(x_test_lr)

# initialize network
batch_count = int(x_train_hr.shape[0] / batch_size)
shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])

generator = Generator(shape).generator()
discriminator = Discriminator(image_shape).discriminator()

adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
generator.compile(loss=vgg_loss, optimizer=adam)
discriminator.compile(loss="binary_crossentropy", optimizer=adam)

shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, channels)
gan = get_gan_network(discriminator, shape, generator, adam)

# train epochs
for e in range(1, epochs+1):
    print ('Epoch %d' % e)
    for batchNum in range(batch_count):
        # images for batch
        batchNums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
        image_batch_hr = x_train_hr[batchNums]
        image_batch_lr = x_train_lr[batchNums]
        
        # generate SR
        generated_images_sr = generator.predict(image_batch_lr)

        # label smoothing
        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
        fake_data_Y = np.random.random_sample(batch_size)*0.2
        
        # train discriminator
        discriminator.trainable = True
        
        # create smoothed labels for discriminator and GAN
        d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
        d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
        gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2

        batchNums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
        image_batch_hr = x_train_hr[batchNums]
        image_batch_lr = x_train_lr[batchNums]
        
        # lock discriminator 
        discriminator.trainable = False
        
        # train generator
        loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
        
    print("Loss HR, Loss LR, Loss GAN")
    print(d_loss_real, d_loss_fake, loss_gan, "\n")

    # save model and/or sample images
    if e == 1 or e % 5 == 0:
        imgOut(e, generator)
    if e % 300 == 0:
        generator.save(genSav % e)
        discriminator.save(discSav % e)
        gan.save(ganSav % e)




#
##----------------------------- FOR TESTING INPUTS -----------------------------
#
#%matplotlib inline
#
## path to image
#me_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--')
#me_path = os.path.join(me_dir, 'cass.jpg')  ## image filename-- CROP TO 124 x 92 px
#
#me = Image.open(me_path)
#me = np.expand_dims(me, axis=0)
#me_hr = np.array(me)
#
## plot hires original
#plt.imshow(me_hr[0])
#plt.axis('off')
#
#me_hr_norm = normalize(me_hr)
#me_lr_norm = lr_images(me_hr_norm, 4)
#
## plot lowres input
#plt.imshow(denormalize(me_lr_norm[0]))
#plt.axis('off')
#
#me_bicubic = upsample(me_lr_norm, 4)
#me_bicubic.shape
#
## plot bicubic interpolation
#plt.imshow(denormalize(me_bicubic[0]))
#plt.axis('off')
#
## load model saves
#from keras.models import load_model
#model_dir = os.path.join('C:\\', 'Users', 'cruze', 'Documents', 'CS767', 'finalProj--', 'saves')
#model_path = os.path.join(model_dir, 'gen_model4500.h5')
#generator = load_model(model_path, custom_objects={'vgg_loss': vgg_loss})
#
## predict SR
#generated_me = generator.predict(me_lr_norm)
#
##plot SR
#plt.imshow(denormalize(generated_me[0]))












