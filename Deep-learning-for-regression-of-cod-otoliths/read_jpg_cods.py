# Error in folder:
# /scratch/disk2/Otoliths/codotoliths_erlend/CodOtholiths-MachineLearning/Savannah_Professional_Practice/2015/70117/nr 04 age_02/IMG_0020.JPG
#_0
#/scratch/disk2/Otoliths/codotoliths_erlend/CodOtholiths-MachineLearning/Savannah_Professional_Practice/2015/70117/Nr03age_02
#/scratch/disk2/Otoliths/codotoliths_erlend/CodOtholiths-MachineLearning/Savannah_Professional_Practice/2015/70331
def read_jpg_cods(B4_input_shape=(380, 380, 3), max_dataset_size=5114, whichExposure='min'):
    #    '''
    #    reads one .jpg file in each folder in structure of folders
    #    returns tensor with images, and 1-1 correspondence with age
    #    '''

    # max_dataset_size = 5156
    # B4_input_shape = (380, 380, 3)
    df_cod = pd.DataFrame(columns=['age', 'path', 'ExposureTime'])
    image_tensor1 = np.empty(shape=(max_dataset_size,) + B4_input_shape)
    image_tensor2 = np.empty(shape=(max_dataset_size,) + B4_input_shape)
    image_tensor3 = np.empty(shape=(max_dataset_size,) + B4_input_shape)

    base_dir = '/gpfs/gpfs0/deep/data/Savannah_Professional_Practice2021_06_10_21/CodOtholiths-MachineLearning/Savannah_Professional_Practice'
    df_cod = pd.DataFrame(columns=['age', 'path', 'ExposureTime'])
    base_dirs_posix = Path(base_dir)

    error_count = 0
    add_count = 0
    for some_year_dir in base_dirs_posix.iterdir():
        count = 0
        if not os.path.isdir(some_year_dir) or "Extra" in str(some_year_dir):
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

                # 2013/70028/Nr01_age05/Thumbs.db
                # 2016/70008/Nr01_age07/Thumbs.db
                if len(onlyfiles) != 6:
                    # print(str(len(onlyfiles)) + '\t' + str( yr_station_codage_path[j] ) + "\t" +'\t'.join(map(str,onlyfiles)))
                    error_count += 1
                else:
                    full_path = [os.path.join(yr_station_codage_path[j], f)
                                 for f in os.listdir(yr_station_codage_path[j])
                                 if os.path.isfile(os.path.join(yr_station_codage_path[j], f))]

                    begin_age = cod_age[j].lower().find('age')
                    # print(cod_age[j])
                    age = cod_age[j][begin_age + 3:begin_age + 5]
                    try:
                        age = int(age)
                    except ValueError:
                        # print(yr_station_codage_path[j])
                        # print(cod_age[j])
                        # print(age)
                        # print(begin_age)
                        age = 0

                    # print(age)

                    full_path.sort()
                    exposures_set = set()
                    exposures_list = []
                    for k in range(0, len(full_path)):  # len(full_path) == 6
                        img = Image.open(full_path[k])
                        exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
                        # print(exif['ExposureTime'])
                        exposures_set.add(exif['ExposureTime'])
                        exposures_list.append(exif['ExposureTime'])

                    # if len(exposures_set) != 3:
                    # print("\t"+str (yr_station_codage_path[j] ) + '\t' + str(exposures_list) )
                    #    continue
                    # else:
                    if len(exposures_list) == 6 and len(exposures_set) == 3:

                        expo_args = np.argsort(exposures_list).tolist()
                        # print( "exposures_list"+str(exposures_list) )
                        # print(" argsort: "+str(expo_args) )

                        numpy_images = [0, 0, 0]
                        file_paths = [0, 0, 0]
                        imgs_added = 0

                        # use if loading to memory
                        """
                        for k in [0,2,4]:
                            img = Image.open( full_path[ expo_args[k] ] ) 
                            pil_img = load_img(full_path[ expo_args[k] ], target_size=B4_input_shape, grayscale=False)
                            array_img = img_to_array(pil_img, data_format='channels_last')

                            numpy_images[imgs_added] = array_img
                            file_paths[imgs_added] = full_path[ expo_args[k] ]
                            imgs_added += 1
                        """

                        if expo_args != [1, 4, 0, 3, 2, 5]:
                            print("exposures_list" + str(exposures_list))
                            print(" argsort: " + str(expo_args))
                            # print(file_paths)

                        if whichExposure == 'min':
                            # use if loading to memory
                            pil_img = load_img(full_path[expo_args[0]], target_size=B4_input_shape, grayscale=False)
                            array_img = img_to_array(pil_img, data_format='channels_last')
                            image_tensor1[add_count] = array_img
                            add_count += 1

                            df_cod = df_cod.append({'age': age, 'path': full_path[expo_args[0]], 'light': 1,
                                                    'ExposureTime': exposures_list[expo_args[0]]}, ignore_index=True)
                        if whichExposure == 'middle':
                            # use if loading to memory
                            pil_img = load_img(full_path[expo_args[2]], target_size=B4_input_shape, grayscale=False)
                            array_img = img_to_array(pil_img, data_format='channels_last')
                            image_tensor1[add_count] = array_img
                            add_count += 1

                            df_cod = df_cod.append({'age': age, 'path': full_path[expo_args[2]], 'light': 2,
                                                    'ExposureTime': exposures_list[expo_args[0]]}, ignore_index=True)
                        if whichExposure == 'max':
                            # use if loading to memory
                            pil_img = load_img(full_path[expo_args[4]], target_size=B4_input_shape, grayscale=False)
                            array_img = img_to_array(pil_img, data_format='channels_last')
                            image_tensor1[add_count] = array_img
                            add_count += 1

                            df_cod = df_cod.append({'age': age, 'path': full_path[expo_args[4]], 'light': 3,
                                                    'ExposureTime': exposures_list[expo_args[0]]}, ignore_index=True)

    print("error_count:" + str(error_count))

    print("add_count:" + str(add_count))

    if whichExposure == 'min':
        return image_tensor1, df_cod
    if whichExposure == 'middle':
        return image_tensor2, df_cod
    if whichExposure == 'max':
        return image_tensor3, df_cod

    return None, None
