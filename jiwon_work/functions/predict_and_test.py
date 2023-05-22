from get_img_tensor_v1 import get_tensor_through_imgs_fn as get_tensor_through_imgs_fn_v1
from get_img_tensor_v2 import get_tensor_through_imgs_fn as get_tensor_through_imgs_fn_v2
from tensorflow.keras.models import load_model as keras_load_model
import matplotlib.pyplot as mat_plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
import csv
import os

# ================== 테스트 파일로 추론해보기 ==================

def get_predict_result_fn(RESIZED_WIDTH, RESIZED_HEIGHT, model, fn_version):

    # TEST_FOLDER_PATH = os.path.join(os.getcwd(), os.pardir, os.pardir, 'open', 'test')
    TEST_FOLDER_PATH = '/content/test/'

    test_img_paths = []
    test_img_names = os.listdir(TEST_FOLDER_PATH)

    for img_name in tqdm(test_img_names):

        img_path = os.path.join(TEST_FOLDER_PATH, img_name)
        test_img_paths.append(img_path)

    # print('test_img_paths :', test_img_paths)

    test_image_tensor = np.empty(0)

    if (fn_version == 'v1'):
        fig = mat_plt.figure()

        # 792개의 이미지 데이터
        test_image_tensor = get_tensor_through_imgs_fn_v1(test_img_paths, RESIZED_HEIGHT, RESIZED_WIDTH, fig)
        test_image_tensor = test_image_tensor / 255

    elif (fn_version == 'v2'):
        test_image_tensor = get_tensor_through_imgs_fn_v2(test_img_paths, RESIZED_HEIGHT, RESIZED_WIDTH)

    print('test_image_tensor.shape :', test_image_tensor.shape)

    isString = type(model) == type('str')
    if isString:
        model = keras_load_model(model)

    predicted_result = model.predict(test_image_tensor)
    print('predicted_result.shape :', predicted_result.shape)
    return predicted_result

def get_test_csv_fn(predicted_result, version_name):

    # 추후에 라벨이 추가될 경우 등 확장성 및 모델의 안정성을 고려하여 총 19가지의 하자 유형의 순서가 변경되지 않도록 직접적으로 리스트 생성
    DEFECT_TYPE_NAMES = ["가구수정", "걸레받이수정", "곰팡이", "꼬임", "녹오염", "들뜸", "면불량", "몰딩수정", "반점", "석고수정",
                        "오염", "오타공", "울음", "이음부불량", "창틀,문틀수정", "터짐", "틈새과다", "피스", "훼손"]

    predicted_index_list = np.argmax(predicted_result, axis=1)
    # print('predicted_index_list :', predicted_index_list)

    predicted_labels = []

    for index in predicted_index_list:
        predicted_labels.append(DEFECT_TYPE_NAMES[index])

    print('predicted_labels :', predicted_labels)

    # 연월일_시간
    now = datetime.today().strftime('%Y%m%d_%H%M%S')
    csv_file_name = f'{version_name}_test_{now}.csv'

    predicted_csv = open(csv_file_name, 'w', newline='', encoding='utf-8')
    wr = csv.writer(predicted_csv)
    wr.writerow(['id', 'label'])

    ids = []
    predicted_label_count = len(predicted_labels)

    for index in range(predicted_label_count):
        ids.append(f'TEST_{str(index).zfill(3)}')

    # [ 'TEST_000', ... ]
    # print('ids :', ids)

    wr.writerows([*zip(ids, predicted_labels)])
    predicted_csv.flush()
    predicted_csv.close()

    return csv_file_name
    
# ============================================================