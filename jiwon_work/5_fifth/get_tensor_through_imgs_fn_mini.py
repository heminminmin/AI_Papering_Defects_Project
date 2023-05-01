from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
import numpy as np

# 객체 순회 및 진행률 표시 용도
# pip3 install tqdm
from tqdm import tqdm

def process_images(image, RESIZED_WIDTH, RESIZED_HEIGHT):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (RESIZED_WIDTH, RESIZED_HEIGHT))
    return image

def get_tensor_through_imgs_fn(img_paths, RESIZED_WIDTH, RESIZED_HEIGHT):

    list_of_tensors = []

    # 3457개의 이미지 데이터
    for img_path in tqdm(img_paths):

        # default
        # color_mode="rgb", 
        img = keras_image.load_img(img_path)

        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=629x658 at 0x1BA13D4F790>
        # print('img :', img)

        # 원본 이미지
        x = keras_image.img_to_array(img)
        # print('x :', x.shape)
        
        normalized_x = process_images(x, RESIZED_WIDTH, RESIZED_HEIGHT) / 255

        list_of_tensors.append(normalized_x)
        
    # print(list_of_tensors)
    return np.array(list_of_tensors, dtype=float)
