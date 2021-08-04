# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:50:17 2021

@author: user
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt
from my_utils import utils_paths


print("[INFO] loading network...")
model = load_model("output\model.model")
mlb = pickle.loads(open("output\model.pickle", "rb").read())

imagePaths = sorted(list(utils_paths.list_images("test_img")))
result = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    output = image.copy()
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # 得到兩個最大的結果
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]


    # 找到對應標籤，寫上去
    for (i, j) in enumerate(idxs):
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))

# 展示
    output = output[:,:,::-1]
    plt.figure(figsize=(7,7))
    plt.imshow(output)
    plt.show()

