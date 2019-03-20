import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from keras.models import load_model


test_path = '../images/test/fer2013/'
save_path = '../images/test_save/fer2013/'
# model_path = '../trained_models/align_model/RAF/RAF_vnet2.378-0.96.hdf5'
model_path = '../trained_models/align_model/fer2013/fer2013_unet.331-0.95.hdf5'

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def file_path(test_path):
    test_files = []
    for folder, subfolders, filenames in os.walk(test_path):
        for filename in filenames:
            if filename.endswith(('.jpg')):
                test_files.append(os.path.join(folder, filename))

    # test_num = len(test_files)
    # y_size, x_size = 48, 48
    # test_data = np.zeros(shape=(test_num, y_size, x_size))
    # for test_arg, test_file in enumerate(test_files):
    #     test_array = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    #     test_array = cv2.resize(test_array, (y_size, x_size))
    #     test_data[test_arg] = test_array
    #
    # test_data = np.expand_dims(test_data, -1)

    return test_files

datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset: ', dataset_name)

    align_model = load_model(model_path, compile=False)
    test_paths = file_path(test_path)
    num = len(test_paths)

    for test_arg, test_path in enumerate(test_paths):
        print('Predicting: {}/{}'.format(test_arg, num))
        img = cv2.imread(test_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_array = cv2.resize(gray, (48, 48))
        test_array = np.expand_dims(test_array, 0)
        test_array = np.expand_dims(test_array, -1)
        test_array = preprocess_input(test_array, False)
        basename = os.path.basename(test_path)
        savepath = os.path.join(save_path, basename)
        pred = align_model.predict(test_array)
        pred = np.squeeze(pred)
        out = np.clip(pred, 0, 1).astype(np.float64)

        seg_img = (img.astype(np.float64) * out[:, :, np.newaxis]).astype(np.uint8)
        plt.figure(figsize=(4, 3))
        plt.subplot("131")
        plt.axis('off')
        plt.imshow(img)
        plt.title('Input')
        plt.subplot("132")
        plt.axis('off')
        plt.imshow(pred.reshape(48, 48) > .5, cmap='gray')
        plt.title('Mask')
        plt.subplot('133')
        plt.axis('off')
        plt.imshow(seg_img)
        plt.title('Prediction')
        plt.savefig(savepath)
        # plt.show()

    print('Finished !')

# test_img_path = '/home/tony/lvhui/process_fer2013/images/3.jpg'
# align_model = load_model(model_path, compile=False)
# img = cv2.imread(test_img_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# test_array = cv2.resize(gray, (48, 48))
# test_array = np.expand_dims(test_array, 0)
# test_array = np.expand_dims(test_array, -1)
# test_array = preprocess_input(test_array, False)
#
# start = time.time()
#
# pred = align_model.predict(test_array)
# end = time.time()
# pred = np.squeeze(pred)
# out = np.clip(pred, 0, 1).astype(np.float64)
#
# seg_img = (img.astype(np.float64) * out[:, :, np.newaxis]).astype(np.uint8)
#
# # plt.imshow(seg_img)
# # plt.axis('off')
# # plt.savefig('segimg4.png')
# #
# # plt.imshow(pred.reshape(48, 48) > .5, cmap='gray')
# # plt.axis('off')
# # plt.savefig('mask4.png')
# #
# # plt.imshow()
#
# # end = time.time()
# print('{} s'.format((end-start)))
#
#
# plt.figure(figsize=(4, 3))
# plt.subplot("131")
# plt.axis('off')
# plt.imshow(img)
# plt.title('Input')
# plt.subplot("132")
# plt.axis('off')
# plt.imshow(pred.reshape(100, 100) > .5, cmap='gray')
# plt.title('Mask')
# plt.subplot('133')
# plt.axis('off')
# plt.imshow(seg_img)
# plt.title('Prediction')
# plt.savefig('result_RAF6.png')
# plt.show()