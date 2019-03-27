from keras.preprocessing.image import ImageDataGenerator
import numpy as np

"""
数据增强代码，进行图片的旋转、伸缩等
"""

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.2,
    horizontal_flip=True,
	fill_mode='nearest')


def data_augment(X_train, y_train):
    X = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2], 1))
    y = np.reshape(y_train,(y_train.shape[0],1))
    for i in range(X.shape[0]):
        x = X[i,:]
        # print(x.shape)
        x = x.reshape((1,) + x.shape) #datagen.flow要求rank为4
        # print(x.shape)
        datagen.fit(x)
        prefix = y[i][0]
        print(prefix)
        counter = 0
        for batch in datagen.flow(x, batch_size=4 , save_to_dir='pic', save_prefix=prefix, save_format='jpg'):
            counter += 1
            if counter > 10:
                break  # 否则生成器会退出循环


if __name__ == '__main__':
    data_augment()
