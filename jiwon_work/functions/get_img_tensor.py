from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.image import per_image_standardization
from tensorflow.image import resize
from tqdm import tqdm
import numpy as np

def get_tensor_through_imgs_fn(img_paths, resized_height, resized_width, fig):

    index = 0
    list_of_tensors = []

    # 3457개의 이미지 데이터
    for img_path in tqdm(img_paths):

        # 1
        img = keras_image.load_img(img_path)

        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=629x658 at 0x1BA13D4F790>
        # print('img :', img)

        # 2
        x = keras_image.img_to_array(img)
        # print('x :', x.shape)

        # 3
        # 평균이 0이고 분산이 1이 되도록 각 이미지의 크기를 선형으로 조정합니다
        standardization_x = per_image_standardization(x)

        # 4
        resized_x = resize(
            standardization_x, [resized_height, resized_width])
        # print('reszied_x: ', resized_x)

        # 5
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

    # 6
    return np.vstack(list_of_tensors)
