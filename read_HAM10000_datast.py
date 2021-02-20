import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image

# Read the CSV that consists of lesion id, image_id, and other details of images
HAM_df  = pd.read_csv('') 

# add images that are present in part 1 and part 2 of the dataset. 
img_path = {os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join('data/HAM10000/', '*', '*.jpg'))}

# now add the path of images to the dataframe
HAM_df['path'] = HAM_df['image_id'].map(img_path.get)
# open image and resize it to 32 and convert into numpy array
HAM_df['image'] = HAM_df['path'].map(lambda x: np.asarray(Image.open(x).resize(32, 32)))

###################################################################################
import shutil

# creating sub folders and adding images in relevant folder

data_dir = os.getcwd() + "/Skin Cancer data"
new_data_dir = os.getcwd() + "/Skin Cancer data/modified data"
HAM_df2 = pd.read_csv('') 

# get the unique labels i.e. classes in the dataset 
labels = HAM_df2['dx'].unique().tolist()
image_labels = []

for i in labels:
    os.mkdir(new_data_dir + str(i) + "/") # make a directory of the label
    sample = HAM_df2[HAM_df2['dx'] == i]['image_id'] # get the image id of the label. We get multiple images
    image_labels.extend(sample)
    for id in image_labels:
        shutil.copyfile((data_dir + '/' + id + "/.jpg")), (new_data_dir + id + "/.jpg")
    images_label = []
    
####################################################################################

# using keras
from keras.preprocessing.image import ImageDataGenerator
import os

datagen = ImageDataGenerator()

train_dir = os.getcwd() + "/Skin Cancer data/modified data" 

train_data = datagen.flow_from_directory(directory = train_dir, 
                                         class_mode = 'categorical',
                                         batch_size = 32,
                                         target_size = (32, 32))   
x, y = next(train_data)

#####################################################################################
# using pytorch
import torchvision
from torchvision import transforms
import torchutils.data as data

train_dir = os.getcwd() + "/Skin Cancer data/modified data" 
transform_img = transforms.Compose([transforms.Resize(32).
                                    transforms.HorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.normalize(mean = [0.5, 0.5, 0.5],
                                                         std = [0.5, 0.5, 0.5])
                                    ])
train_data_pyt = torchvision.datasets.ImageFolder(root = train_dir)

        




