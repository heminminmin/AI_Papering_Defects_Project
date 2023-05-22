from tensorflow.image import per_image_standardization as tf_per_image_standardization
from tensorflow.keras.preprocessing.image import img_to_array as keras_img_to_array
from tensorflow.keras.preprocessing.image import load_img as keras_load_img
from tensorflow.image import resize as tf_resize
from tqdm import tqdm
import numpy as np

def process_images(image, resized_width, resized_height):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf_per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf_resize(image, (resized_width, resized_height))
    return image

def get_tensor_through_imgs_fn(img_paths, resized_width, resized_height):

    list_of_tensors = []

    # 3457개의 이미지 데이터
    for img_path in tqdm(img_paths):

        # 1
        # default
        # color_mode="rgb", 
        img = keras_load_img(img_path)

        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=629x658 at 0x1BA13D4F790>
        # print('img :', img)

        # 2
        x = keras_img_to_array(img)
        # print('x :', x.shape)
        
        # 3, 4
        normalized_x = process_images(x, resized_width, resized_height) / 255

        list_of_tensors.append(normalized_x)
        
    # print(list_of_tensors)

    # 5
    return np.array(list_of_tensors, dtype=float)
