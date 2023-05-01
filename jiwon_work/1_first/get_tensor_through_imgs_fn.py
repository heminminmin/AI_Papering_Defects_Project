from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf
import numpy as np

# 객체 순회 및 진행률 표시 용도
# pip3 install tqdm
from tqdm import tqdm

def get_tensor_through_imgs_fn(img_paths, resized_height, resized_width, fig):

    index = 0
    list_of_tensors = []

    # 3457개의 이미지 데이터
    for img_path in tqdm(img_paths):

        img = keras_image.load_img(img_path)

        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=629x658 at 0x1BA13D4F790>
        # print('img :', img)

        # 원본 이미지
        x = keras_image.img_to_array(img)
        # print('x :', x.shape)

        # 정규화된 이미지
        standardization_x = tf.image.per_image_standardization(x)

        # 크기가 조정된 이미지
        resized_x = tf.image.resize(
            standardization_x, [resized_height, resized_width])
        # print('reszied_x: ', resized_x)

        expanded_x = np.expand_dims(resized_x, axis=0)
        # print('new_x :', expanded_x.shape)

        list_of_tensors.append(expanded_x)

        # nrows 장수만큼만 이미지 확인해보기
        nrows = 3
        ncols = 3
        if index < nrows:

            ax1 = fig.add_subplot(nrows, ncols, index *
                                 ncols + 1, xticks=[], yticks=[])
            ax1.imshow(img)

            ax2 = fig.add_subplot(nrows, ncols, index *
                                 ncols + 2, xticks=[], yticks=[])
            ax2.imshow(standardization_x)

            ax3 = fig.add_subplot(nrows, ncols, index *
                                 ncols + 3, xticks=[], yticks=[])
            ax3.imshow(resized_x)

        index += 1

    return np.vstack(list_of_tensors)
