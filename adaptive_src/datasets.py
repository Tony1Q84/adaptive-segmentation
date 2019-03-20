import numpy as np
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

def traindata(mask_source_path):
    mask_paths =[]

    for folder, subfolders, filenames in os.walk(mask_source_path):
        for filename in filenames:
            if filename.endswith(('.jpg')):
                mask_paths.append(os.path.join(folder, filename))

    num_data = len(mask_paths)
    y_size, x_size = 64, 64
    datas = np.zeros(shape=(num_data, y_size, x_size))
    masks = np.zeros(shape=(num_data, y_size, x_size))
    for mask_arg, mask_path in enumerate(mask_paths):
        data_path = mask_path.replace('RAF_mask', 'aligned')
        mask_array = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_array = cv2.resize(mask_array, (y_size, x_size))
        data_array = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        data_array = cv2.resize(data_array, (y_size, x_size))
        datas[mask_arg] = data_array
        masks[mask_arg] = mask_array

    datas = np.expand_dims(datas, -1)
    masks = np.expand_dims(masks, -1)

    return datas, masks

# imgs, masks = traindata('../datasets/fer2013_crop/')
# print('imgs.shape: ', imgs.shape)
# print('masks.shape: ', masks.shape)