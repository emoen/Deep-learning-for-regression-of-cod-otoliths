# Source: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
# tf_efficientnetv2_s_in21k - input_size=(3, 300, 300), test_input_size=(3, 384, 384)
# tf_efficientnetv2_m_in21k - input_size=(3, 384, 384), test_input_size=(3, 480, 480)
# tf_efficientnetv2_l_in21k - input_size=(3, 384, 384), test_input_size=(3, 480, 480)
# tf_efficientnetv2_xl_in21k -input_size=(3, 384, 384), test_input_size=(3, 512, 512)

class CONFIG:
    seed = 42
    torch_model_name = 'tf_efficientnetv2_xl_in21k'
    train_batch_size = 16
    valid_batch_size = 8
    img_size = 456
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
    epochs = 1 #50
    early_stopping_patience = 14
    reduceLROnPlateau_factor = 0.2
    reduceLROnPlateau_patience = 7
    base_dir = '/gpfs/gpfs0/deep/data/Savannah_Professional_Practice2021_08_12_2021/CodOtholiths-MachineLearning/Savannah_Professional_Practice'