import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def regress_area():
    pd_xy = pd.DataFrame(None, columns=['age', 'pixels'])

    dir1 = '2017'
    dir2 = '2018'

    pathlist = Path(dir1).glob('**/*.JPG')
    for path in pathlist:
         # because path is object not string
         img_path = str(path)
         #print(img_path)
         age_idx = img_path.find('age')
         #print(age_idx)
         age = img_path[age_idx+3:age_idx+5]
         age = int(age)
         #print("age:"+str(age))
         pixels = get_pixels(img_path)
         pd_xy=pd_xy.append({'age':age, 'size':pixels}, ignore_index=True)
         
    X = pd_xy.pixels.values
    y = pd_xy.age.values
    model = LinearRegression()
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    X_test=[]
    y_pred =[]
    for i, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X.iloc[train,:], y.iloc[train,:])
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(mse)
        score = model.score(X.iloc[test,:], y.iloc[test,:])
        scores.append(score)
        
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
        
    print(scores)
         
         
def get_pixels(img):
    gray = cv2.imread(img,0)
    
    th, im_th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY);
    
    if type(im_th) == type(None):
        print("img:"+str(img)+" not an image")
        return 1
        
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, None, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    
    n_white_pix = np.sum(im_out == 255)
    
    return n_white_pix

    #cv2.imwrite('area_oto.png',im_out)
    
if __name__ == '__main__':
    regress_area()
