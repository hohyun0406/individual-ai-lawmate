from datasets import load_dataset

# 데이터셋 로드 (예: csv 파일)
dataset = load_dataset('csv', data_files='C:/Users/bit/Ideaproject/individual-ml-lawmate/ml-work-space/4model/train_data.csv')['train']

# 데이터셋의 컬럼 확인
print(dataset.column_names)
