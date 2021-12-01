import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image, ExifTags

from tensorflow.keras.preprocessing.image import img_to_array, load_img


def read_jpg_cods2(config):
    """
    reads a .jpg file in each folder in structure of folders
    depending on light exposure in {min, middle, max}
    returns tensor with images, and 1-1 correspondence with age
    """
    df_cod = pd.DataFrame(columns=['age', 'image', 'path', 'light', 'ExposureTime'])

    base_dir = '/gpfs/gpfs0/deep/data/Savannah_Professional_Practice2021_06_10_21/CodOtholiths-MachineLearning/Savannah_Professional_Practice'
    base_dirs_posix = Path(base_dir)

    error_count = 0
    add_count = 0
    for some_year_dir in base_dirs_posix.iterdir():
        if config.debugging: # terminate quickly for testing
            if add_count > 0:
                break

        if not os.path.isdir(some_year_dir) or "Extra" in str(some_year_dir): #dont read files in root dir, or folder "Extra"
            continue

        # dir structure: /year/station_number/cod_img_by_age/6 jpeg images of one fish
        stat_nos = [name for name in os.listdir(some_year_dir) if os.path.isdir(os.path.join(some_year_dir, name))]
        for i in range(0, len(stat_nos)):
            cod_path = os.path.join(some_year_dir, stat_nos[i])
            yr_station_codage_path = [os.path.join(cod_path, n) for n in os.listdir(cod_path)
                                      if os.path.isdir(os.path.join(cod_path, n))]
            cod_age = [n for n in os.listdir(cod_path)
                       if os.path.isdir(os.path.join(cod_path, n))]

            assert len(yr_station_codage_path) == len(cod_age)
            for j in range(0, len(yr_station_codage_path)):
                # print(onlyfiles)
                onlyfiles = [f for f in os.listdir(yr_station_codage_path[j])
                             if os.path.isfile(os.path.join(yr_station_codage_path[j], f))]

                if len(onlyfiles) != 6:
                    # print(str(len(onlyfiles)) + '\t' + str( yr_station_codage_path[j] ) + "\t" +'\t'.join(map(str,onlyfiles)))
                    error_count += 1
                else:
                    full_path = [os.path.join(yr_station_codage_path[j], f)
                                 for f in os.listdir(yr_station_codage_path[j])
                                 if os.path.isfile(os.path.join(yr_station_codage_path[j], f))]

                    begin_age = cod_age[j].lower().find('age')
                    age = cod_age[j][begin_age + 3:begin_age + 5]
                    try:
                        age = int(age)
                    except ValueError:
                        age = 0
                        continue

                    full_path.sort()
                    exposures_set = set()
                    exposures_list = []
                    for k in range(0, len(full_path)):
                        img = Image.open(full_path[k])
                        exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
                        exposures_set.add(exif['ExposureTime'])  # requires: 3 unique exposures,
                        exposures_list.append(exif['ExposureTime'])  # requires: 6 exposures total - rotated 180 deg

                    if len(exposures_list) == 6 and len(exposures_set) == 3 and age not in [0, 14, 15, 16, 17]:
                        expo_args = np.argsort(exposures_list).tolist()

                        # print( "exposures_list"+str(exposures_list) )
                        # print(" argsort: "+str(expo_args) )
                        # if expo_args != [1, 4, 0, 3, 2, 5]:
                        # print( "exposures_list"+str(exposures_list) )
                        # print(" argsort: "+str(expo_args) )

                        index_to_exposed_jpg = -1
                        light_value = -1
                        if config.which_exposure == 'min':
                            index_to_exposed_jpg = 0
                            light_value = 1
                        if config.which_exposure == 'middle':
                            index_to_exposed_jpg = 2
                            light_value = 2

                        if config.which_exposure == 'max':
                            index_to_exposed_jpg = 4
                            light_value = 3

                        pil_img = load_img(full_path[expo_args[ index_to_exposed_jpg ]], target_size=(config.img_size, config.img_size))
                        array_img = img_to_array(pil_img, data_format=config.CHANNELS)
                        add_count += 1
                        print("fp:"+str(full_path[expo_args[ index_to_exposed_jpg ]]) )
                        df_cod = df_cod.append({
                            'age': age,
                            'image': array_img,
                            'path': full_path[expo_args[ index_to_exposed_jpg ]],
                            'light': light_value,
                            'ExposureTime': exposures_list[expo_args[ index_to_exposed_jpg ]]}, ignore_index=True)

    print("error_count:" + str(error_count))
    print("add_count:" + str(add_count))
    return df_cod