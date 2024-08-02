from trl import SFTTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# 데이터셋 불러오기
train_df = pd.read_csv('C:/Users/bit/Ideaproject/ml-work-space/4model/train_data.csv')
val_df = pd.read_csv('C:/Users/bit/Ideaproject/ml-work-space/4model/val_data.csv')

# 모델과 토크나이저 준비
model_name = "bert-base-uncased"  # 사용할 BERT 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 클래스 수에 맞춰 조정

# SFT Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_df,
    eval_dataset=val_df,
    tokenizer=tokenizer,
    training_args={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 5e-5,
        "logging_dir": "./logs",
        "output_dir": "./results"
    }
)

# 모델 훈련
trainer.train()