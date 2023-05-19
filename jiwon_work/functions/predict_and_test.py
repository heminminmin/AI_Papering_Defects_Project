from get_img_tensor import get_tensor_through_imgs_fn
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
import csv
import os

# pip3 install pandas
import pandas as pd

# ================== 테스트 파일로 추론해보기 ==================

def get_predict_result_fn(RESIZED_WIDTH, RESIZED_HEIGHT, model):

    TEST_FOLDER_PATH = os.path.join(os.getcwd(), os.pardir, os.pardir, 'open', 'test')

    test_img_paths = []
    test_img_names = os.listdir(TEST_FOLDER_PATH)

    for img_name in tqdm(test_img_names):

        img_path = os.path.join(TEST_FOLDER_PATH, img_name)
        test_img_paths.append(img_path)

    # print('test_img_paths :', test_img_paths)

    fig = plt.figure()

    # 792개의 이미지 데이터
    test_image_tensor = get_tensor_through_imgs_fn(test_img_paths, RESIZED_HEIGHT, RESIZED_WIDTH, fig)
    normalized_test_image_tensor = test_image_tensor / 255
    print('normalized_test_image_tensor.shape :',
        normalized_test_image_tensor.shape)

    isString = type(model) == type('str')
    if isString:
        model = load_model(model)

    predicted_result = model.predict(normalized_test_image_tensor)
    print('predicted_result.shape :', predicted_result.shape)
    return predicted_result

def get_test_csv_fn(predict_result):

    # 추후에 라벨이 추가될 경우 등 확장성 및 모델의 안정성을 고려하여 총 19가지의 하자 유형의 순서가 변경되지 않도록 직접적으로 리스트 생성
    DEFECT_TYPE_NAMES = ["가구수정", "걸레받이수정", "곰팡이", "꼬임", "녹오염", "들뜸", "면불량", "몰딩수정", "반점", "석고수정",
                        "오염", "오타공", "울음", "이음부불량", "창틀,문틀수정", "터짐", "틈새과다", "피스", "훼손"]

    predicted_index_list = np.argmax(predict_result, axis=1)
    # print('predicted_index_list :', predicted_index_list)

    predicted_labels = []

    for index in predicted_index_list:
        predicted_labels.append(DEFECT_TYPE_NAMES[index])

    print('predicted_labels :', predicted_labels)

    # 연월일_시간
    now = datetime.today().strftime('%Y%m%d_%H%M%S')
    predicted_csv = open(f'test_{now}.csv', 'w', newline='', encoding='utf-8')
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
    
# ============================================================