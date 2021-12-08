from .train_val_test_split import train_validate_test_split #import from relative path in /util
import numpy as np


def train_test_split(df, config, test_size=0.15, a_seed=8):
    train_idx, val_idx, test_idx = train_validate_test_split(
        range(0, len(df)),
        validation_set_size=0.1,
        test_set_size=test_size,
        a_seed=a_seed)

    train_idx_oof = train_idx + val_idx  # prepare - train+validation set for KFold cv split
    val_idx = None

    train_imgs = np.empty(shape=(len(train_idx_oof),) + config.input_shape)  # shape=(len(train_idx),)+B4_input_shape)
    train_imgs = np.stack(df.iloc[train_idx_oof].image, axis=0)
    train_age = df.iloc[train_idx_oof].age.values

    test_imgs = np.empty(shape=(len(test_idx),) + config.input_shape)
    test_imgs = np.stack(df.iloc[test_idx].image, axis=0)
    test_age = df.iloc[test_idx].age.values

    test_path = df.iloc[test_idx]
    test_path = test_path.copy(deep=True)
    test_path = test_path.drop(columns=['image'])
    test_path.reset_index(drop=True, inplace=True)

    # test_path.to_csv('test_set_files.csv', columns=['path', 'age', 'light', 'ExposureTime'], index=False)

    if config.debugging:
        print("train:" + str(train_imgs.shape))
        print("trainType:" + str(type(train_imgs)))
        print("train_age:" + str(train_age.shape))
        print("train_ageType" + str(type(train_age)))

        print("test:" + str(test_imgs.shape))
        print("testType:" + str(type(test_imgs)))
        print("test_age:" + str(test_age.shape))
        print("test_ageType" + str(type(test_age)))

    train_imgs = np.multiply(train_imgs, 1. / 255)  # Train- + Validation images
    test_imgs = np.multiply(test_imgs, 1. / 255)

    return train_imgs, train_age, test_imgs, test_age, test_path