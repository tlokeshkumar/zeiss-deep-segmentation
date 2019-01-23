from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

N_BANDS = 1
N_CLASSES = 1  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [1.0]
N_EPOCHS = 1000
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 32
TRAIN_SZ = BATCH_SIZE*100  # train size
VAL_SZ = 2    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = str(1).zfill(2)  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    img_m1 = np.expand_dims(normalize(tiff.imread('./data/mband/{}.tif'.format('01'))),-1)
    mask1 = np.expand_dims(tiff.imread('./data/gt_mband/{}.tif'.format('01')) / 255,-1)
    img_m2 = np.expand_dims(normalize(tiff.imread('./data/mband/{}.tif'.format('02'))),-1)
    mask2 = np.expand_dims(tiff.imread('./data/gt_mband/{}.tif'.format('02')) / 255,-1)    
    img_m = np.concatenate([img_m1,img_m2],0)[:280,:,:,:]    
    mask = np.concatenate([mask1,mask2],0)[:280,:,:,:]

    imgs = img_m[-1,:,:,:]

    # train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
    print(img_m.shape)

    for i in range(280):
        X_DICT_TRAIN[str(i)] = img_m[i,:,:,:]
        Y_DICT_TRAIN[str(i)] = mask[i,:,:,:]
    # X_DICT_VALIDATION[trainIds] = img_m[train_xsz:, :, :]
    # Y_DICT_VALIDATION[trainIds] = mask[train_xsz:, :, :]

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN,Y_DICT_TRAIN  , n_patches=TRAIN_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard])
        return model

    train_net()
