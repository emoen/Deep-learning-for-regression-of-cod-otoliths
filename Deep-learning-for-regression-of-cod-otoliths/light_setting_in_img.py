base = "/scratch/disk2/Otoliths/endrem/deep/data/Savannah_Professional_Practice2021/2012/70303/Nr01_age07/"

img = Image.open(base+"IMG_0025.JPG")
exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
#>>> exif['ExposureTime']
#0.01


img = Image.open("IMG_0026.JPG")
exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
#>>> exif['ExposureTime']
#0.005

img = Image.open("IMG_0027.JPG")
exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
#>>> exif['ExposureTime']
#0.02
