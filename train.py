# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:29:17 2021

@author: user
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from model_name.simplevgg import SimpleVGGNet
import matplotlib.pyplot as plt
from my_utils import utils_paths
import numpy as np
import random
import pickle
import cv2
import os


# 超參數初始化
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# 獲取數據路徑
print("[INFO] 加載數據...")
imagePaths = sorted(list(utils_paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []

# 獲取數據標籤，為了讀取多標籤。ex:red,shirt
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	data.append(image)
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)

# 預處理
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# 制作標籤
print("[INFO] 數據標籤:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# 打印標籤代表
for (i, label) in enumerate(mlb.classes_): #長度為6 (3個顏色3種服飾)、 ex:[011000] 、每個位置都作2分判斷。1.black 2.blue 3.dress 4.jeans 5.red 6.shirt
	print("{}. {}".format(i + 1, label))

# 數據集切分
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# 數據增強
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# 最終是依次二分
print("[INFO] compiling model...")
model = SimpleVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")#2分類用sigmoid

# Adam動態調整學習率
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# 損失函數
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 訓練網絡
print("[INFO] 訓練網路...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# 保存模型
print("[INFO] 保存...")
model.save("output\model.model")

plot_model(model, to_file='model.png',show_shapes=True)
# 保存標籤
print("[INFO] serializing label binarizer...")
f = open("output\model.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()


plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot1.PNG")