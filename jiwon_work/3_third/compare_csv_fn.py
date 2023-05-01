import pandas as pd
import os

# ======== 기존 csv 파일과 데이터 비교해보기 ========

def compare_with_sample_csv_fn(CURRENT_CSV_FILE_NAME):

    # 디폴트 : '훼손'
    SAMPLE_CSV_FILE_NAME = 'sample_submission.csv'
    sample_csv = pd.read_csv(os.path.join(os.path.pardir, os.path.pardir, 'open', SAMPLE_CSV_FILE_NAME))
    curr_csv = pd.read_csv(os.path.join(os.path.pardir, 'third', CURRENT_CSV_FILE_NAME))

    count = 0
    total_label_count = len(sample_csv['label'])

    for index in range(total_label_count):

        is_changing = sample_csv['label'][index] != curr_csv['label'][index]
        if is_changing:
            count += 1

    # 달라진 라벨의 개수
    print('compare_with_sample_csv_fn() :', count)

def compare_with_prev_csv_fn(PREV_CSV_FILE_NAME, CURRENT_CSV_FILE_NAME):

    # count : 250
    # PREV_CSV_FILE_NAME = 'test_20230429_023516.csv'

    # count : 322
    # PREV_CSV_FILE_NAME = 'test_20230429_101620.csv'

    prev_csv = pd.read_csv(os.path.join(os.path.pardir, 'first', PREV_CSV_FILE_NAME))
    curr_csv = pd.read_csv(os.path.join(os.path.pardir, 'third', CURRENT_CSV_FILE_NAME))

    count = 0
    total_label_count = len(prev_csv['label'])

    for index in range(total_label_count):

        is_changing = prev_csv['label'][index] != curr_csv['label'][index]
        if is_changing:
            print(prev_csv['label'][index], curr_csv['label'][index])
            count += 1

    # 달라진 라벨의 개수
    print('compare_with_prev_csv_fn() :', count)

# ==================================================