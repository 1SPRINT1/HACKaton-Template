import tkinter as tk
from tkinter import filedialog
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter

# КЛАССЫ КОТОРЫЕ РАСПОЗНАЁТ НЕЙРОСЕТЬ И РАЗМЕРЫ ИЗОБРАЖЕНИЙ
CLASSES = 8

COLORS = ['black', 'red', 'lime',
          'blue', 'orange', 'pink',
          'cyan', 'magenta']

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (512, 512)

# СОЗДАНИЕ СЛОЁВ НЕЙРОСЕТИ
def input_layer():
    return tf.keras.layers.Input(shape=SAMPLE_SIZE + (3,))

def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample_block(filters, size, dropout=False):
    initializer = tf.keras.initializers.GlorotNormal()

    result = tf.keras.Sequential()

    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                        kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())
    return result

def output_layer(size):
    initializer = tf.keras.initializers.GlorotNormal()
    return tf.keras.layers.Conv2DTranspose(CLASSES, size, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='sigmoid')

# СОЗДАНИЕ АРХИТЕКТУРЫ НЕЙРОННОЙ СЕТИ
inp_layer = input_layer()

downsample_stack = [
    downsample_block(64, 4, batch_norm=False),
    downsample_block(128, 4),
    downsample_block(256, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
    downsample_block(512, 4),
]

upsample_stack = [
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(512, 4, dropout=True),
    upsample_block(256, 4),
    upsample_block(128, 4),
    upsample_block(64, 4)
]

out_layer = output_layer(4)

# Реализуем skip connections
x = inp_layer

downsample_skips = []

for block in downsample_stack:
    x = block(x)
    downsample_skips.append(x)

downsample_skips = reversed(downsample_skips[:-1])

for up_block, down_block in zip(upsample_stack, downsample_skips):
    x = up_block(x)
    x = tf.keras.layers.Concatenate()([x, down_block])

out_layer = out_layer(x)

unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)

#КНОПКИ
def select_folder_photo():
    folder_photo = filedialog.askdirectory()
    print("Папка с выбранными фотографиями:")
    print(folder_photo)
    return(folder_photo)

def select_folder():
    folder = filedialog.askdirectory()
    print("Выбранная папка для сохранения:")
    print(folder)
    return(folder)

def select_net_weights():
    folder_with_net_weights = filedialog.askdirectory()
    print("Выбранная папка с весами нейросети:")
    print(folder_with_net_weights)
    return(folder_with_net_weights)

def start_predict():
    folder_photo = select_folder_photo() #Папка с выбранными фотографиями
    folder = select_folder() #папка для сохранения
    folder_with_net_weights = select_net_weights() #папка с весами нейросети

    # ЗАГРУЗКА ВЕСОВ МОДЕЛИ
    unet_like.load_weights(f'{folder_with_net_weights}/unet_like')
    # В ПАПКЕ Networks ДОЛЖНЫ БЫТЬ Веса нейросети

    # ОБРАБОТКА ФОТОГРАФИЙ
    rgb_colors = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 165, 0),
        (255, 192, 203),
        (0, 255, 255),
        (255, 0, 255)
    ]

    frames = sorted(glob.glob(f'{folder_photo}/*.png'))

    for filename in frames:
        frame = imread(filename)
        sample = resize(frame, SAMPLE_SIZE)

        predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))
        predict = predict.reshape(SAMPLE_SIZE + (CLASSES,))

        scale = frame.shape[0] / SAMPLE_SIZE[0], frame.shape[1] / SAMPLE_SIZE[1]

        frame = (frame / 1.1).astype(np.uint8)

        for channel in range(1, CLASSES):
            contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
            contours = measure.find_contours(np.array(predict[:, :, channel]))

            try:
                for contour in contours:
                    rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                               contour[:, 1] * scale[1],
                                               shape=contour_overlay.shape)

                    contour_overlay[rr, cc] = 1

                contour_overlay = dilation(contour_overlay, disk(1))
                frame[contour_overlay == 1] = rgb_colors[channel]
            except:
                pass

        imsave(f'{folder}/{os.path.basename(filename)}', frame)
        # Папка куда сохраняются обработанные фотографии

root = tk.Tk()
root.title("Сегментация изображений")
root.geometry('600x600')
root['bg'] = 'green'

instruction_label = tk.Label(root, text="Как нажмёте на кнопку Начать распознование\nследуйте следующим шагам:", font='Arial 16 bold',
                              bg='DarkGoldenrod1', activebackground='DarkGoldenrod3')

instruction_label.pack()
instruction_label.place(y=20, x=60)

select_photo_label = tk.Label(root, text="1) Выбрать папку с фото", font='Arial 16 bold',
                              bg='DarkGoldenrod1', activebackground='DarkGoldenrod3')

select_photo_label.pack()
select_photo_label.place(y=100, x=150)

select_folder_label = tk.Label(root, text="2) Выбрать папку для\nсохранения изображений", font='Arial 16 bold',
                               bg='DarkGoldenrod1', activebackground='DarkGoldenrod3')
select_folder_label.pack()
select_folder_label.place(y=140, x=145)

select_folder_NW_label = tk.Label(root, text="3) Выбрать папку с весами нейросети",font='Arial 16 bold',
                                  bg='DarkGoldenrod1', activebackground='DarkGoldenrod3')
select_folder_NW_label.pack()
select_folder_NW_label.place(y=205, x=85)

attention_label = tk.Label(root, text="Как вы выполните шаг №3\nнейросеть начнёт обрабатывать изображения\nЭто может занять немного времени", font='Arial 16 bold',
                              bg='DarkGoldenrod1', activebackground='DarkGoldenrod3')

attention_label.pack()
attention_label.place(y=260, x=50)

start_predict_button = tk.Button(root, text="Начать распознавание", command=start_predict,
                                    font='Arial 19 bold', bg='orange', activebackground='DarkGoldenrod3')
start_predict_button.pack()
start_predict_button.place(y=500, x=135)

root.mainloop()