# Source: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
# tf_efficientnetv2_s_in21k - input_size=(3, 300, 300), test_input_size=(3, 384, 384)
# tf_efficientnetv2_m_in21k - input_size=(3, 384, 384), test_input_size=(3, 480, 480)
# tf_efficientnetv2_l_in21k - input_size=(3, 384, 384), test_input_size=(3, 480, 480)
# tf_efficientnetv2_xl_in21k -input_size=(3, 384, 384), test_input_size=(3, 512, 512)

class CONFIG:
    seed = 42
    CHANNELS = 'channels_last' #'channels_first' or 'channels_last' - pytorch or keras convention - needed by PIL
    which_exposure = 'min'
    debugging = False
    train_batch_size = 8
    valid_batch_size = 8
    img_size = 384
    val_img_size = 480
    epochs = 1000
    learning_rate = 1e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    n_accumulate = 1
    n_fold = 5
    target_size = 1