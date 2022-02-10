import matplotlib.pyplot as plt
import numpy as np

from utils.read_jpg_cods import read_jpg_cods
from utils.train_val_test_split import train_validate_test_split
from utils.train_test_split import *

plt.rcParams['figure.dpi'] = 350

class CONFIG:
    debugging= False
    img_size = 384
    base_dir = '/gpfs/gpfs0/deep/data/Savannah_Professional_Practice2021_08_12_2021/CodOtholiths-MachineLearning/Savannah_Professional_Practice'
    which_exposure = "min"
    CHANNELS = "channels_first"
    input_shape = (3, img_size, img_size)
    test_size = 0.1
    test_split_seed = 8


df = read_jpg_cods( CONFIG ) 

age = df.age
unique, counts = np.unique(age, return_counts=True)
age_range_freq = dict(zip(unique, counts))
print("Age dist")
print(age_range_freq)

N_points = len(age)
n_bins = 2
x=unique
y=counts
plt.bar(x, height=y)
plt.show()

##### TEST #######
train_imgs, train_age, test_imgs, test_age, test_path = train_test_split(df, CONFIG, test_size=CONFIG.test_size, a_seed=CONFIG.test_split_seed)

unique, counts = np.unique(test_age, return_counts=True)
age_range_freq = dict(zip(unique, counts))
print("Test age dist")
print(age_range_freq)

N_points = len(age)
n_bins = 2
x=unique
y=counts
plt.bar(x, height=y)
plt.show()
