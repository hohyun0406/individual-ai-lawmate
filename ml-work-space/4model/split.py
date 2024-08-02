from sklearn.model_selection import train_test_split
import pandas as pd

# 전처리된 데이터 로드
df = pd.read_csv('C:/Users/bit/Ideaproject/ml-work-space/2predata/combined_preprocessed_data.csv')

# train과 validation 데이터로 나누기
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 파일로 저장
train_df.to_csv('C:/Users/bit/Ideaproject/ml-work-space/4model/train_data.csv', index=False, encoding='utf-8-sig')
val_df.to_csv('C:/Users/bit/Ideaproject/ml-work-space/4model/val_data.csv', index=False, encoding='utf-8-sig')
