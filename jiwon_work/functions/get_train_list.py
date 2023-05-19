import os
from tensorflow.keras.utils import to_categorical

# ======== 학습용 이미지 경로와 라벨에 대한 각각의 리스트 생성하기 ========

# c:\Users\UserK\Desktop\hansol_J\jiwon_work\..\..\open\train
TRAIN_FOLDER_PATH = os.path.join(os.getcwd(), os.pardir, os.pardir, 'open', 'train')
# TRAIN_FOLDER_PATH = '/content/train/'
# print('TRAIN_FOLDER_PATH : ', TRAIN_FOLDER_PATH)

def get_train_image_paths_and_encoding_labels_fn():

    defect_types = os.listdir(TRAIN_FOLDER_PATH)
    # print('defect_types :', defect_types)

    labels = []
    train_image_paths = []

    for defect_type in defect_types:

        file_names = os.listdir(os.path.join(TRAIN_FOLDER_PATH, defect_type))
        # print('file_names: ', file_names)

        for file_name in file_names:
            train_image_paths.append(os.path.join(TRAIN_FOLDER_PATH, defect_type, file_name))
            labels.append(defect_type)

    # ====================== 라벨 정규화 과정 ======================

    # 문자열 형태의 범주형 데이터인 라벨 데이터를 정수 형태로 정규화
    normalized_labels = []

    # 추후에 라벨이 추가될 경우 등 확장성 및 모델의 안정성을 고려하여 총 19가지의 하자 유형의 순서가 변경되지 않도록 직접적으로 리스트 생성
    DEFECT_TYPE_NAMES = ["가구수정", "걸레받이수정", "곰팡이", "꼬임", "녹오염", "들뜸", "면불량", "몰딩수정", "반점", "석고수정",
                        "오염", "오타공", "울음", "이음부불량", "창틀,문틀수정", "터짐", "틈새과다", "피스", "훼손"]

    defect_type_count = len(DEFECT_TYPE_NAMES)

    for label in labels:
        normalized_labels.append(DEFECT_TYPE_NAMES.index(label))

    # [ 0, 0, 0, ... ]
    # print('normalized_labels :', normalized_labels)

    # 정수형 클래스의 레이블을 이진 클래스의 원핫 인코딩 벡터로 변환
    encoding_labels = to_categorical(normalized_labels, defect_type_count)
    print('encoding_labels.shape :', encoding_labels.shape)

    # =============================================================

    return train_image_paths, encoding_labels

# [ 하자 유형에 대한 폴더명\\파일명.확장자, ... ]
# print('train_image_paths: ', train_image_paths)

# [ 하자 유형에 대한 폴더명, ... ]
# print('encoding_labels :', encoding_labels)

# =====================================================================