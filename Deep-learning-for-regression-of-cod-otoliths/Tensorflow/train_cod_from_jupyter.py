import pandas as pd
import numpy as np
import os
from pathlib import Path
from PIL import Image, ExifTags
import gc #garbage collection

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
import scipy

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback

#cod utils
from utils.read_jpg_cods import *
from utils.train_val_test_split import *
from utils.train_test_split import *

# if pretraining with salmon-scales
#from salmon_scales.train_salmon_scale_util import read_images, load_xy, get_checkpoint_tensorboard, create_model_grayscale, get_fresh_weights, base_output, dense1_linear_output, train_validate_test_split

#from salmon_scales import train_salmon_scale_util

def base_output(model):
    z = model.output
    z = GlobalMaxPooling2D()(z)
    return z

#def custom_activation(x):
#    return (1/(1 + K.exp(-x)))

#get_custom_objects().update({'custom_activation': Activation(custom_activation)})

###model.add(keras.layers.Dense(128, activation="custom_activation", input_shape=(784,)))


def dense1_linear_output(gray_model):
    z = base_output(gray_model)
    z = Dense(1, activation='linear')(z)
    #z = tf.keras.layers.Lambda( lambda x: 2.8*tf.math.tanh(x, name="output_sigmoid") )(z)
    return z


def get_checkpoint_tensorboard(tensorboard_path, checkpoint_path):
    tensorboard = TensorBoard(log_dir=tensorboard_path)
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True, save_weights_only=False)
    return tensorboard, checkpointer


def base_model(input_shape):
    ############# Define model ################
    data_augmentation = tf.keras.Sequential([
      layers.experimental.preprocessing.RandomFlip("horizontal"),
      layers.experimental.preprocessing.RandomRotation(factor=(-0.5, 0.5))],
      name="img_augmentation",
      # possible width shift, height shift
    )

    rgb_efficientNetB = tf.keras.applications.EfficientNetB4(include_top=False, weights='imagenet',
                                                              input_shape=input_shape, classes=2)
    z = dense1_linear_output( rgb_efficientNetB )
    cod_tmp = Model(inputs=rgb_efficientNetB.input, outputs=z)

    inputLayer = tf.keras.layers.Input(shape=input_shape)
    x = data_augmentation(inputLayer)
    x = cod_tmp(x)
    cod = Model(inputs=inputLayer, outputs=x)

    print( cod.summary() )
    return cod

def binary_accuracy_for_regression(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def compile_model( cod, config ):
    ########### Compile model ##################
    learning_rate = config.learning_rate #0.00001  # 0.00005
    adam = optimizers.Adam(learning_rate=learning_rate)

    for layer in cod.layers:
        layer.trainable = True

    cod.compile(loss='mse', optimizer=adam, metrics=['mse', binary_accuracy_for_regression])
    return cod

def do_train(df, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICE
    tensorboard_path = config.ROOTDIR + config.tensorboard_path
    checkpoint_path = config.ROOTDIR +config.checkpoint_path
    ############## pretrain salmon_scales ###############
    """
    cod = base_model(config.input_shape) B4_input_shape
    cod = compile_model(cod, config)
    new_shape = config.input_shape

    print("loadxy")
    rb_imgs, all_sea_age, all_smolt_age, all_farmed_class, all_spawn_class, all_filenames = load_xy()

    uten_ukjent = len(all_sea_age) - all_sea_age.count(-1.0)
    rb_imgs2 = np.empty(shape=(uten_ukjent,)+new_shape)
    unique, counts = np.unique(all_sea_age, return_counts=True)
    print("age distrib:"+str( dict(zip(unique, counts)) ))

    all_sea_age2 = []
    found_count = 0
    all_filenames2 = []
    for i in range(0, len(all_sea_age)):
        if all_sea_age[i] > -1:
            rb_imgs2[found_count] = rb_imgs[i]
            all_sea_age2.append(all_sea_age[i])
            found_count += 1
            all_filenames2.append(all_filenames[i])

    assert found_count == uten_ukjent

    age_scales = all_sea_age2
    rb_imgs = rb_imgs2
    age_scales = np.vstack(age_scales)

    train_datagen_scales = ImageDataGenerator(
        zca_whitening=False,
        width_shift_range=0.2,
        height_shift_range=0.2, #20,
        #zoom_range=[0.5,1.0],
        rotation_range=360,
        horizontal_flip=False,
        vertical_flip=True,
        rescale=1./255)

    #Tensorflow 2.2 - wrap generator in tf.data.Dataset
    def callGen():
        return train_datagen_scales.flow( rb_imgs, age_scales, batch_size=config.train_batch_size )

    train_dataset = tf.data.Dataset.from_generator(callGen, (tf.float32, tf.float32)).shuffle(128, reshuffle_each_iteration=True).repeat()

    history_callback_scales = cod.fit(train_dataset,
        steps_per_epoch=1000,
        epochs=20,
        #callbacks=[early_stopper, tensorboard, checkpointer],
        #validation_data= (val_rb_imgs, val_age),
        class_weight=None)

    cod.save("NNSalmonScales/salmonNNmodel.h5")
    print("Saved salmon model to disk")

    print("len img tensor"+str(image_tensor.shape))
    """
    ####### End train cod ##################################
    ####### Train/Test split ################################
    train_imgs, train_age, test_imgs, test_age, test_path = train_test_split(df, config, test_size=config.test_size, a_seed=config.test_split_seed)
    test_path.to_csv( config.ROOTDIR+"test_set_files.csv", index=False)
    ######### Run KFold ####################
    test_predictions_nn = np.zeros(test_age.shape[0])

    the_fold = 0
    a_seed = config.KERAS_TRAIN_TEST_SEED #2021
    numberOfFolds = config.n_fold #5
    kfold = StratifiedKFold(n_splits=numberOfFolds, random_state=a_seed, shuffle=True)
    #kfold = KFold(n_splits = 5, random_state = a_seed, shuffle = True)
    #for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_age)):
    print("len train_imgs, len train_age:"+str(train_imgs.shape)+" "+str(len(train_age)))
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_imgs, train_age.tolist())):
        train_imgs_new = train_imgs[train_idx]
        train_age_new = train_age[train_idx]
        val_imgs_new = train_imgs[val_idx]
        val_age_new = train_age[val_idx]

        """ standard scalar """
        unique, counts = np.unique( train_age_new, return_counts=True)
        print( dict(zip(unique, counts)) )
        scaler = StandardScaler()
        scaler.fit( train_age_new.reshape(-1,1) )
        train_age_new = scaler.transform( train_age_new.reshape(-1,1) ).squeeze()
        val_age_new   = scaler.transform( val_age_new.reshape(-1,1) ).squeeze()
        test_age_new  = scaler.transform( test_age.reshape(-1,1) ).squeeze()

        print("scalar transform:")
        print( train_age_new[0:20] )
        print("min/max train_age:"+str(np.min(np.asarray(train_age)))+ ", "+ str(np.max(np.asarray(train_age))) )
        print("min/max train_age:"+str(np.min(np.asarray(train_age_new)))+ ", "+ str(np.max(np.asarray(train_age_new))) )
        print("min/max val_age:"+str(np.min(np.asarray(val_age_new)))+ ", "+ str(np.max(np.asarray(val_age_new))) )
        print("min/max val_age_new:"+str(np.min(np.asarray(val_age_new)))+ ", "+ str(np.max(np.asarray(val_age_new))) )
        print("min/max test_age:"+str(np.min(np.asarray(test_age)))+ ", "+ str(np.max(np.asarray(test_age))) )
        print("min/max test_age_new:"+str(np.min(np.asarray(test_age_new)))+ ", "+ str(np.max(np.asarray(test_age_new))) )
        print("train_age_new shape:"+str(train_age_new.shape))
        print("val_age_new shape:"+str(val_age_new.shape))
        print("test_age_new shape:"+str(test_age_new.shape))
        """ end standard scalar """

        #####################################################
        train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs_new, train_age_new))
        train_dataset = train_dataset.shuffle(len(train_age_new))
        train_dataset = train_dataset.batch(config.train_batch_size)
        train_dataset = train_dataset.repeat(1000)

        ######### build, compile model ####################
        cod = base_model(config.input_shape)
        #cod = compile_model(cod, config)
        #cod = tf.keras.models.load_model("NNSalmonScales/salmonNNmodel.h5",
        #        custom_objects={"binary_accuracy_for_regression":binary_accuracy_for_regression},
        #        compile=False)
        cod = compile_model(cod, config)

        tensorboard, checkpointer = get_checkpoint_tensorboard(tensorboard_path, checkpoint_path)

        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience,
                                                         verbose=0, mode = 'min', restore_best_weights = True)
        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=config.reduceLROnPlateau_factor,
                                                       patience=config.reduceLROnPlateau_patience,
                                                       verbose = 0, mode = 'min')

        log_file_name = config.ROOTDIR + 'loss_log/loss_log2_'+str(the_fold)+'.txt'
        print( "log_file_name:"+str(log_file_name) )
        txt_log = open(log_file_name, mode='wt', buffering=1)

        save_op_callback = LambdaCallback(
          on_epoch_end = lambda epoch, logs: txt_log.write(
            str( {'epoch': epoch, 'loss': logs['loss']} ) + '\n'),
          on_train_end = lambda logs: txt_log.close()
        )

        K.set_value(cod.optimizer.learning_rate, config.learning_rate)
        print("Learning rate before second fit:", cod.optimizer.learning_rate.numpy())

        history_callback = cod.fit(train_dataset,
                                   steps_per_epoch=config.steps_per_epoch,
                                   epochs=config.epochs,
                                   callbacks=[early_stopper, plateau, tensorboard,
                                       checkpointer, save_op_callback],
                                   validation_data=(val_imgs_new, val_age_new),  # (val_rb_imgs, val_age),
                                   class_weight=None)

        test_metrics = cod.evaluate(x=test_imgs, y=test_age_new)  # np.vstack(test_age)
        print("test metric:" + str(cod.metrics_names))
        print("test metrics:" + str(test_metrics))
        f = open(config.ROOTDIR + "test_metric_"+str(the_fold)+".txt", "w")
        f.write("test metric:\n")
        f.write(str(cod.metrics_names))
        f.write("\n")
        f.write(str(test_metrics))
        f.close()

        y_pred_test = cod.predict(test_imgs, verbose=1)
        print("shape of prediction:"+str(y_pred_test.shape))
        print("scaled predictions+"+str( y_pred_test[0:20] ) )
        y_pred_test = scaler.inverse_transform( y_pred_test )
        print("invers_transf pred:" + str( y_pred_test[0:20] ) )
        print("y_pred_test.shape:"+str( y_pred_test.shape ) )
        y_pred_test = np.squeeze( y_pred_test )
        y_pred_test = y_pred_test.tolist()
        y_pred_test_rounded = [int(round(x)) for x in y_pred_test]

        print("test_age_new.shape"+str( test_age.shape ))
        test_age_rounded = test_age.tolist() #np.squeeze( test_age_new )
        test_age_rounded = [int(round(x)) for x in test_age_rounded]

        test_predictions_nn += np.asarray( y_pred_test ) / numberOfFolds

        ## aggregate accuracy so far: on fold 2: {2/5, 4/5, 6/5} * 5/2
        test_predictions_nn_mse = test_predictions_nn * ( numberOfFolds/(the_fold+1) )
        test_predictions_nn_rounded = [int(round(x)) for x in test_predictions_nn_mse]

        mse = mean_squared_error(test_age, y_pred_test )
        acc = accuracy_score( test_age_rounded, y_pred_test_rounded )
        acc_agg = accuracy_score( test_age_rounded, test_predictions_nn_rounded )
        mse_agg = mean_squared_error( test_age, test_predictions_nn_mse )
        print("acc:"+str( acc ) )
        print("acc_agg:"+str( acc_agg ) )
        print("mse:"+str( mse ) )
        print("mse_agg:"+str( mse_agg ) )

        ######### Write test predictions to file ##########
        df_test_statistics = pd.DataFrame()
        df_test_statistics['y_pred_test'] = y_pred_test
        df_test_statistics['y_pred_test_rounded'] = y_pred_test_rounded
        df_test_statistics['y_true'] = test_age_rounded
        df_test_statistics['mse'] = np.abs(df_test_statistics['y_pred_test']-df_test_statistics['y_true'])**2
        df_test_statistics['acc'] = df_test_statistics['y_pred_test_rounded'] == df_test_statistics['y_true']
        df_test_statistics['acc'] = df_test_statistics['acc'].astype(int)
        df_test_statistics['test_predictions_nn'] = test_predictions_nn
        df_test_statistics['test_predictions_nn_rounded'] = test_predictions_nn_rounded
        df_test_statistics['agg_mse'] = np.abs(df_test_statistics['test_predictions_nn']-df_test_statistics['y_true'])**2
        df_test_statistics['agg_acc'] = df_test_statistics['test_predictions_nn_rounded'] == df_test_statistics['y_true']
        df_test_statistics['agg_acc'] = df_test_statistics['agg_acc'].astype(int)

        df_test_statistics['agg_stat'] = 0
        df_test_statistics.at[0, 'agg_stat'] = acc
        df_test_statistics.at[1, 'agg_stat'] = acc_agg
        df_test_statistics.at[2, 'agg_stat'] = mse
        df_test_statistics.at[3, 'agg_stat'] = mse_agg
        df_test_statistics['y_pred_test'] = y_pred_test

        df_test_statistics.to_csv(config.ROOTDIR+'test_set_'+str(the_fold)+'.csv', index=False)
        ###### Save model #############
        # serialize weights to HDF5
        cod.save(config.ROOTDIR+"EffNetB4/EffNetB4_{}.h5".format( the_fold ))
        print("Saved model to disk")
        the_fold += 1
        ###### garbage collect ############
        del train_imgs_new, train_age_new
        del val_imgs_new, val_age_new
        del cod
        # statistics
        del test_metrics
        del y_pred_test
        del test_age_rounded, test_predictions_nn_rounded
        del df_test_statistics
        gc.collect()

    return test_predictions_nn, np.squeeze( test_age )


class CONFIG:
    seed = 42
    torch_model_name = 'tf_efficientnetv2_xl_in21k'
    train_batch_size = 16
    valid_batch_size = 8
    img_size = 10 #456
    val_img_size = 480
    epochs = 1000
    learning_rate = 1e-5
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = 10
    scheduler = 'CosineAnnealingLR'
    n_accumulate = 1
    n_fold = 5
    target_size = 1
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    debugging = False
    which_exposure = 'min'
    CHANNELS = 'channels_last' #'channels_first' or 'channels_last' - pytorch or keras convention - needed by PIL
    KERAS_TRAIN_TEST_SEED = 2021
    ROOTDIR = "./EFFNetB5_groupkfold_stdScalar/"
    CUDA_VISIBLE_DEVICE = "0"
    tensorboard_path = 'tensorboard_test2'
    checkpoint_path = 'checkpoints_test2/cod_oto_efficientnetBBB.{epoch:03d}-{val_loss:.2f}.hdf5'
    input_shape = (img_size, img_size, 3)
    test_size = 0.15
    test_split_seed = 8
    steps_per_epoch = 1600
    epochs = 150
    early_stopping_patience = 14
    reduceLROnPlateau_factor = 0.2
    reduceLROnPlateau_patience = 7
    base_dir = '/gpfs/gpfs0/deep/data/Savannah_Professional_Practice2021_08_12_2021/CodOtholiths-MachineLearning/Savannah_Professional_Practice'
    #base_dir = '/scratch/disk2/Otoliths/endrem/deep/data/Savannah_Professional_Practice2021_08_12_2021/CodOtholiths-MachineLearning/Savannah_Professional_Practice'

import json
def dump_config_to_json( CONFIG ):
    config_dict = CONFIG.__dict__
    config_dict = dict(config_dict)
    config_dict.pop('device', None)
    config_dict.pop('__dict__', None)
    config_dict.pop('__weakref__', None)
    config_dict.pop('__doc__', None)
    with open(CONFIG.ROOTDIR + 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    dump_config_to_json(CONFIG)
    df_cod = read_jpg_cods( CONFIG ) #5316
    print("len df:"+str( len(df_cod) ) )

    test_predictions_nn, test_age = do_train(df_cod, CONFIG)

    print("test predictions:\n"+str( test_predictions_nn ) )
    print("test_age:\n"+str(test_age) )

    mse = mean_squared_error(test_age, test_predictions_nn)
    print("mse:"+str( mse) )

    pred_rounded = [int(round(x)) for x in test_predictions_nn]
    print("test age rounded shape:"+str(test_age.shape))
    test_age_rounded = np.squeeze( test_age )
    print("test age rounded shape:"+str(test_age_rounded.shape))
    test_age_rounded = [int(round(x)) for x in test_age_rounded]
    acc = accuracy_score(test_age_rounded, pred_rounded )

    print("acc:"+str( acc ) )

    df_final = pd.DataFrame()
    df_final['test_pred_nn'] = test_predictions_nn
    df_final['y_true'] = test_age
    df_final.to_csv(CONFIG.ROOTDIR+'pred_mean_final.csv', index=False)
