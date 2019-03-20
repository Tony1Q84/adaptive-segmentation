from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from align_unet import unet, tiny_unet, test_AE, vnet, vnet_V2, vnet_V3
from datasets import traindata
from datasets import preprocess_input

batch_size = 32
num_epochs = 1000
verbose = 1
patience = 100
train_mask_path = '../datasets/RAF/Image/RAF_mask/'
log_path = '../trained_models/align_model/RAF/'

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')

data_gen_args = dict()
image_datagen = ImageDataGenerator(**data_gen_args)
# mask_datagen = ImageDataGenerator(**data_gen_args)

model = vnet_V3()
datasets = ['RAF']

for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callback
    log_file_path = log_path + dataset_name + '_align_vnet2_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('loss', factor=0.1,
                                  patience=int(patience/10), verbose=1)
    trained_models_path = log_path + dataset_name +'_vnet2'
    model_names = trained_models_path + '.{epoch:02d}-{acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'loss', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    datas, masks = traindata(train_mask_path)
    datas = preprocess_input(datas, False)
    masks = preprocess_input(masks, False)
    masks[masks > 0.5] = 1
    masks[masks <= 0.5] = 0

    # seed = 1
    # image_generator = image_datagen.flow(datas, seed=seed, batch_size=batch_size)
    # mask_generator = mask_datagen.flow(masks, seed=seed, batch_size=batch_size)
    # train_generator = zip(image_generator, mask_generator)
    model.fit_generator(image_datagen.flow(datas, masks, batch_size),
                        steps_per_epoch=len(datas) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks)
    # model.fit_generator(train_generator,
    #                     steps_per_epoch=len(datas) / batch_size,
    #                     epochs=num_epochs, verbose=1, callbacks=callbacks)