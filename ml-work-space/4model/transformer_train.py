import pandas as pd
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# 데이터셋 로드
train_df = pd.read_csv('C:/Users/bit/Ideaproject/ml-work-space/4model/train_data.csv')
val_df = pd.read_csv('C:/Users/bit/Ideaproject/ml-work-space/4model/val_data.csv')

# 데이터셋을 Hugging Face Dataset 형식으로 변환
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# BERT 토크나이저와 모델 불러오기
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# 토큰화 함수
def tokenize_function(examples):
    return tokenizer(examples['question'], examples['answer'], truncation=True)

# 데이터셋 토큰화
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',          # 결과를 저장할 디렉토리
    evaluation_strategy="epoch",     # 평가 전략
    learning_rate=2e-5,              # 학습률
    per_device_train_batch_size=8,   # 훈련 배치 크기
    per_device_eval_batch_size=8,    # 검증 배치 크기
    num_train_epochs=3,              # 훈련 에폭 수
    weight_decay=0.01,               # 가중치 감소
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 모델 훈련 시작
trainer.train()
